use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

use async_trait::async_trait;
use serde_json::Value;

use crate::tools::{ToolCallArg, ToolCallFn};

struct Skill {
    pub header: SkillHeader,
    pub content: String,
    pub directory: PathBuf,
}

impl Skill {
    pub fn load(path: PathBuf) -> Option<Self> {
        let directory = path.clone();
        let skill_path = path.join("SKILL.md");
        let skill_content = fs::read_to_string(skill_path).ok()?;
        let mut split = skill_content.splitn(3, "---");
        _ = split.next();
        let header_str = split.next()?;
        let header = SkillHeader::try_from(header_str).ok()?;
        let content = split.next()?.to_string();
        Some(Self {
            header,
            content,
            directory,
        })
    }
}

/// Skills as described per the spec at `https://agentskills.io/specification`.
struct SkillHeader {
    /// Max 64 characters. Lowercase letters, numbers, and hyphens only. Must not start or end with a hyphen.
    pub name: String,
    /// Max 1024 characters. Non-empty. Describes what the skill does and when to use it.
    pub description: String,
    /// License name or reference to a bundled license file.
    pub license: Option<String>,
    /// Max 500 characters. Indicates environment requirements (intended product, system packages, network access, etc.).
    pub compatibility: Option<String>,
    /// Arbitrary key-value mapping for additional metadata.
    pub metadata: Option<HashMap<String, String>>,
    /// Space-separated string of pre-approved tools the skill may use. (Experimental)
    // TODO: the spec is really vague about this so I will omit what exactly this does for now
    pub allowed_tools: Option<String>,
}

#[derive(Clone, Debug)]
enum SkillHeaderParseError {
    InvalidKeyValFormat,
    MissingFields(&'static str),
    NameTooLong,
    DescriptionTooLong,
    EmptyDescription,
    InvalidName,
    CompatTooLong,
    Unknown,
}

impl TryFrom<&str> for SkillHeader {
    type Error = SkillHeaderParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut in_metadata = false;
        let mut name: Option<&str> = None;
        let mut description: Option<&str> = None;
        let mut license: Option<&str> = None;
        let mut compatibility: Option<&str> = None;
        let mut allowed_tools: Option<&str> = None;
        let mut metadata: HashMap<&str, &str> = HashMap::new();

        for line in value.lines() {
            let Some((mut key, mut value)) = line.split_once(':') else {
                return Err(SkillHeaderParseError::InvalidKeyValFormat);
            };
            key = key.trim();
            value = value.trim();

            match key {
                "metadata" => {
                    in_metadata = true;
                }
                "name" => {
                    name.replace(value);
                }
                "description" => {
                    description.replace(value);
                }
                "license" => {
                    license.replace(value);
                }
                "compatibility" => {
                    compatibility.replace(value);
                }
                "allowed_tools" => {
                    allowed_tools.replace(value);
                }
                _ if in_metadata => {
                    metadata.insert(key, value);
                }
                _ => {}
            }
        }
        let Some(name) = name else {
            return Err(SkillHeaderParseError::MissingFields("name"));
        };
        let Some(description) = description else {
            return Err(SkillHeaderParseError::MissingFields("description"));
        };

        if name.chars().count() > 64 {
            return Err(SkillHeaderParseError::NameTooLong);
        }

        if name.starts_with('-')
            || name
                .chars()
                .any(|c| !matches!(c, '-' | 'a'..'z' | '0'..'9'))
        {
            return Err(SkillHeaderParseError::InvalidName);
        }

        if description.chars().count() > 1024 {
            return Err(SkillHeaderParseError::DescriptionTooLong);
        }

        if description.is_empty() {
            return Err(SkillHeaderParseError::EmptyDescription);
        }

        if let Some(compat) = compatibility.as_ref()
            && compat.chars().count() > 500
        {
            return Err(SkillHeaderParseError::CompatTooLong);
        }

        let metadata = if metadata.is_empty() {
            None
        } else {
            Some(
                metadata
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v.to_owned()))
                    .collect(),
            )
        };

        Ok(SkillHeader {
            name: name.to_string(),
            description: description.to_string(),
            license: license.map(ToString::to_string),
            compatibility: compatibility.map(ToString::to_string),
            metadata,
            allowed_tools: allowed_tools.map(ToString::to_string),
        })
    }
}

struct SkillsDirectory(PathBuf);

#[async_trait]
impl ToolCallFn for SkillsDirectory {
    fn get_args(&self) -> Vec<ToolCallArg> {
        vec![ToolCallArg::string(
            "name",
            "the name of the skill to load.",
        )]
    }

    fn get_description(&self) -> &'static str {
        "Load skill files from the skill directory."
    }

    fn get_name(&self) -> &'static str {
        "load_skills"
    }

    async fn invoke(&self, args: &serde_json::Value) -> String {
        let Some(arg) = args.get("name") else {
            return "Please provide a skill name to load.".to_string();
        };
        let Value::String(arg) = arg else {
            return "The skill has to be a string".to_string();
        };

        let skills_path = self.0.clone();
        let Ok(skill) = get_skill(skills_path, arg) else {
            return "Failed loading that skill".to_string();
        };
        skill
    }

    fn get_timeout_wait(&self) -> Duration {
        Duration::ZERO
    }
}

#[allow(unused)]
fn get_skill<P>(mut skills_path: PathBuf, path: P) -> Result<String, io::Error>
where
    P: AsRef<Path>,
{
    skills_path.push(path);
    skills_path.push("SKILL.md");
    fs::read_to_string(skills_path)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use crate::skills::{SkillHeader, SkillHeaderParseError, get_skill};

    #[test]
    fn parse_full_skill_header() {
        let header = r#"name: test-skill
description: this thing does something
metadata:
    foo: thing
    bar: some other thing
license: thing"#;
        let header = SkillHeader::try_from(header).unwrap();
        assert_eq!(header.name, "test-skill");
        assert_eq!(header.description, "this thing does something");
        assert!(header.metadata.is_some());
        assert!(header.license.is_some());
        assert!(header.compatibility.is_none())
    }

    #[test]
    fn parse_minimal_header() {
        let header = "name: foo\ndescription: bar";
        let header = SkillHeader::try_from(header).unwrap();
        assert_eq!(header.name, "foo");
        assert_eq!(header.description, "bar");
        assert!(header.license.is_none());
        assert!(header.compatibility.is_none());
        assert!(header.metadata.is_none());
        assert!(header.allowed_tools.is_none());
    }

    #[test]
    fn parse_all_optional_fields() {
        let header = "name: foo\ndescription: bar\nlicense: MIT\ncompatibility: linux\nallowed_tools: bash grep";
        let header = SkillHeader::try_from(header).unwrap();
        assert_eq!(header.license.as_deref(), Some("MIT"));
        assert_eq!(header.compatibility.as_deref(), Some("linux"));
        assert_eq!(header.allowed_tools.as_deref(), Some("bash grep"));
    }

    #[test]
    fn parse_metadata_collected() {
        let header = "name: foo\ndescription: bar\nmetadata:\n    foo: thing\n    bar: other";
        let header = SkillHeader::try_from(header).unwrap();
        let metadata = header.metadata.expect("metadata should be present");
        assert_eq!(metadata.get("foo").map(|s| s.as_str()), Some("thing"));
        assert_eq!(metadata.get("bar").map(|s| s.as_str()), Some("other"));
    }

    #[test]
    fn parse_values_trimmed() {
        let header = "name:    foo   \ndescription:    bar   ";
        let header = SkillHeader::try_from(header).unwrap();
        assert_eq!(header.name, "foo");
        assert_eq!(header.description, "bar");
    }

    #[test]
    fn parse_missing_name() {
        let header = "description: bar";
        let parsed = SkillHeader::try_from(header);
        assert!(matches!(
            parsed,
            Err(SkillHeaderParseError::MissingFields("name"))
        ));
    }

    #[test]
    fn parse_missing_description() {
        let header = "name: foo";
        let parsed = SkillHeader::try_from(header);
        assert!(matches!(
            parsed,
            Err(SkillHeaderParseError::MissingFields("description"))
        ));
    }

    #[test]
    fn parse_invalid_key_val_format() {
        let header = "name: foo\ndescriptionno colon here";
        let parsed = SkillHeader::try_from(header);
        assert!(matches!(
            parsed,
            Err(SkillHeaderParseError::InvalidKeyValFormat)
        ));
    }

    #[test]
    fn parse_name_too_long() {
        let name = "a".repeat(65);
        let header = format!("name: {name}\ndescription: bar");
        let parsed = SkillHeader::try_from(header.as_str());
        assert!(matches!(parsed, Err(SkillHeaderParseError::NameTooLong)));
    }

    #[test]
    fn parse_name_exactly_64_chars_ok() {
        let name = "a".repeat(64);
        let header = format!("name: {name}\ndescription: bar");
        SkillHeader::try_from(header.as_str()).unwrap();
    }

    #[test]
    fn parse_description_too_long() {
        let desc = "a".repeat(1025);
        let header = format!("name: foo\ndescription: {desc}");
        let parsed = SkillHeader::try_from(header.as_str());
        assert!(matches!(
            parsed,
            Err(SkillHeaderParseError::DescriptionTooLong)
        ));
    }

    #[test]
    fn parse_description_exactly_1024_chars_ok() {
        let desc = "a".repeat(1024);
        let header = format!("name: foo\ndescription: {desc}");
        SkillHeader::try_from(header.as_str()).unwrap();
    }

    #[test]
    fn parse_empty_description() {
        let header = "name: foo\ndescription:";
        let parsed = SkillHeader::try_from(header);
        assert!(matches!(
            parsed,
            Err(SkillHeaderParseError::EmptyDescription)
        ));
    }

    #[test]
    fn parse_compat_too_long() {
        let compat = "a".repeat(501);
        let header = format!("name: foo\ndescription: bar\ncompatibility: {compat}");
        let parsed = SkillHeader::try_from(header.as_str());
        assert!(matches!(parsed, Err(SkillHeaderParseError::CompatTooLong)));
    }

    #[test]
    fn parse_compat_exactly_500_chars_ok() {
        let compat = "a".repeat(500);
        let header = format!("name: foo\ndescription: bar\ncompatibility: {compat}");
        SkillHeader::try_from(header.as_str()).unwrap();
    }

    #[test]
    fn parse_invalid_name_starts_with_hyphen() {
        let header = "name: -foo\ndescription: bar";
        let parsed = SkillHeader::try_from(header);
        assert!(matches!(parsed, Err(SkillHeaderParseError::InvalidName)));
    }

    #[test]
    fn parse_invalid_name_uppercase() {
        let header = "name: Foo\ndescription: bar";
        let parsed = SkillHeader::try_from(header);
        assert!(matches!(parsed, Err(SkillHeaderParseError::InvalidName)));
    }

    #[test]
    fn parse_invalid_name_underscore() {
        let header = "name: foo_bar\ndescription: bar";
        let parsed = SkillHeader::try_from(header);
        assert!(matches!(parsed, Err(SkillHeaderParseError::InvalidName)));
    }

    #[test]
    fn parse_valid_name_with_hyphen_in_middle() {
        let header = "name: foo-bar\ndescription: bar";
        let parsed = SkillHeader::try_from(header).unwrap();
        assert_eq!(parsed.name, "foo-bar");
    }

    #[test]
    fn parse_valid_name_with_numbers() {
        let header = "name: foo123\ndescription: bar";
        let parsed = SkillHeader::try_from(header).unwrap();
        assert_eq!(parsed.name, "foo123");
    }

    #[test]
    fn parse_unknown_keys_before_metadata_ignored() {
        let header = "name: foo\ndescription: bar\nsome_unknown: value\nextra: stuff";
        let parsed = SkillHeader::try_from(header).unwrap();
        assert_eq!(parsed.name, "foo");
    }

    fn make_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "skills-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn get_skill_reads_skill_md() {
        let dir = make_temp_dir();
        let skill_dir = dir.join("my-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        let skill_file = skill_dir.join("SKILL.md");
        fs::write(&skill_file, "# My Skill\ncontent").unwrap();

        let result = get_skill(dir.clone(), "my-skill").unwrap();
        assert_eq!(result, "# My Skill\ncontent");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_skill_missing_returns_error() {
        let dir = make_temp_dir();
        let result = get_skill(dir.clone(), "nope");
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_skill_pushes_subpath() {
        let dir = make_temp_dir();
        let skill_dir = dir.join("nested").join("skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), "deep").unwrap();

        let result = get_skill(dir.clone(), Path::new("nested/skill")).unwrap();
        assert_eq!(result, "deep");

        let _ = fs::remove_dir_all(&dir);
    }
}
