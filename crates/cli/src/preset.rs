//! Presets: built-in TOML files plus `--set path.to.key=value` overrides.
//!
//! Layering, lowest to highest priority:
//! 1. Rust `Default` impls (every field has one),
//! 2. the preset file (built-in name or a path),
//! 3. `--set` overrides,
//! 4. dedicated CLI flags such as `--target-lufs` (applied in `main`).
//!
//! Overrides are merged into the TOML *document* before it is turned into
//! structs, so they get exactly the same type checking and unknown-field
//! rejection as the file itself.

use anyhow::{Context, Result, bail};
use dsp::config::{PostConfig, PreConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;

const BUILTIN: [(&str, &str); 2] = [
    ("youtube", include_str!("../presets/youtube.toml")),
    ("podcast", include_str!("../presets/podcast.toml")),
];

/// Preset files larger than this are refused (a preset is a few KB).
const MAX_PRESET_BYTES: u64 = 1 << 20;

/// What to do when the ML stage cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fallback {
    Rnnoise,
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RestoreSettings {
    pub clean_model: String,
    pub studio_model: String,
    pub fallback: Fallback,
    pub timeout_s: u64,
}

impl Default for RestoreSettings {
    fn default() -> Self {
        Self {
            clean_model: "deepfilternet3".into(),
            studio_model: "sidon".into(),
            fallback: Fallback::Rnnoise,
            timeout_s: 900,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Preset {
    pub restore: RestoreSettings,
    pub pre: PreConfig,
    pub post: PostConfig,
}

impl Preset {
    /// Load a built-in preset by name, or a TOML file by path, then apply
    /// `key=value` overrides.
    pub fn load(name_or_path: &str, overrides: &[String]) -> Result<Self> {
        let text = match BUILTIN.iter().find(|(name, _)| *name == name_or_path) {
            Some((_, text)) => (*text).to_owned(),
            None => read_preset_file(Path::new(name_or_path))?,
        };
        let mut doc: toml::Table = toml::from_str(&text)
            .with_context(|| format!("preset `{name_or_path}` is not valid TOML"))?;
        for item in overrides {
            apply_override(&mut doc, item)?;
        }
        toml::Value::Table(doc)
            .try_into::<Preset>()
            .with_context(|| format!("preset `{name_or_path}` (with --set overrides) is invalid"))
    }

    pub fn builtin_names() -> impl Iterator<Item = &'static str> {
        BUILTIN.iter().map(|(name, _)| *name)
    }
}

fn read_preset_file(path: &Path) -> Result<String> {
    let meta = std::fs::metadata(path).with_context(|| {
        let names: Vec<_> = Preset::builtin_names().collect();
        format!(
            "`{}` is neither a built-in preset ({}) nor a readable file",
            path.display(),
            names.join(", ")
        )
    })?;
    if meta.len() > MAX_PRESET_BYTES {
        bail!(
            "preset file {} is {} bytes; refusing anything over 1 MiB",
            path.display(),
            meta.len()
        );
    }
    std::fs::read_to_string(path).with_context(|| format!("cannot read preset {}", path.display()))
}

/// Apply one `a.b.c=value` override. The value is parsed as a TOML value
/// (`4`, `-1.5`, `true`, `"60"`), falling back to a plain string so
/// `--set pre.hum.mode=off` works without quotes.
fn apply_override(doc: &mut toml::Table, item: &str) -> Result<()> {
    let (path, raw) = item
        .split_once('=')
        .with_context(|| format!("--set `{item}`: expected key=value"))?;
    let keys: Vec<&str> = path.trim().split('.').collect();
    if keys.iter().any(|k| k.is_empty()) {
        bail!("--set `{item}`: empty key segment");
    }
    let value = parse_value(raw.trim());

    let (last, parents) = keys.split_last().expect("split always yields one item");
    let mut table = doc;
    for key in parents {
        table = table
            .entry(key.to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .with_context(|| format!("--set `{item}`: `{key}` is a value, not a section"))?;
    }
    table.insert(last.to_string(), value);
    Ok(())
}

fn parse_value(raw: &str) -> toml::Value {
    // Parse `x = <raw>` so TOML decides the type.
    toml::from_str::<toml::Table>(&format!("x = {raw}"))
        .ok()
        .and_then(|mut t| t.remove("x"))
        .unwrap_or_else(|| toml::Value::String(raw.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_presets_parse_and_validate() {
        for name in Preset::builtin_names() {
            let p = Preset::load(name, &[]).unwrap();
            p.pre.validate(48_000.0).unwrap();
            p.post.validate(48_000.0).unwrap();
        }
        assert_eq!(
            Preset::load("podcast", &[])
                .unwrap()
                .post
                .loudness
                .target_lufs,
            -16.0
        );
        // youtube.toml spells out the defaults, so it must equal Default.
        assert_eq!(Preset::load("youtube", &[]).unwrap(), Preset::default());
    }

    #[test]
    fn overrides_are_typed_and_checked() {
        let p = Preset::load(
            "youtube",
            &[
                "post.compressor.ratio=4".into(),
                "pre.hum.mode=off".into(),
                "pre.hum.mode=\"60\"".into(),
            ],
        )
        .unwrap();
        assert_eq!(p.post.compressor.ratio, 4.0);
        assert_eq!(p.pre.hum.mode, dsp::config::HumMode::Hz60);

        assert!(Preset::load("youtube", &["post.compressor.ratoi=4".into()]).is_err());
        assert!(Preset::load("youtube", &["post.compressor.ratio=loud".into()]).is_err());
        assert!(Preset::load("youtube", &["post.compressor.ratio.x=1".into()]).is_err());
        assert!(Preset::load("youtube", &["noequals".into()]).is_err());
    }

    #[test]
    fn unknown_preset_name_is_a_clear_error() {
        let err = Preset::load("no-such-preset", &[]).unwrap_err().to_string();
        assert!(err.contains("youtube"), "{err}");
    }
}
