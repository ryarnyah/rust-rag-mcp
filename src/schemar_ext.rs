use rmcp::schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use std::ops::Deref;
use std::ops::DerefMut;

/// A transparent wrapper that generates `anyOf` with a `null` branch.
#[derive(Debug, Clone)]
pub struct Nullable<T>(pub Option<T>);

impl<T> serde::Serialize for Nullable<T>
where
    T: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T> serde::Deserialize<'de> for Nullable<T>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Nullable)
    }
}

impl<T: JsonSchema> JsonSchema for Nullable<T> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        format!("Nullable_{}", T::schema_name()).into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let inner = generator.subschema_for::<T>();
        json_schema!({
            "anyOf": [
                inner,
                { "type": "null" }
            ]
        })
    }
}

impl<T> Deref for Nullable<T> {
    type Target = Option<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Nullable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<Option<T>> for Nullable<T> {
    fn from(v: Option<T>) -> Self {
        Nullable(v)
    }
}
impl<T> From<Nullable<T>> for Option<T> {
    fn from(v: Nullable<T>) -> Self {
        v.0
    }
}

/// An absent field deserializes to `None` (callers then apply their own
/// default) — `#[serde(default)]` on a `Nullable` field needs this.
impl<T> Default for Nullable<T> {
    fn default() -> Self {
        Nullable(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Nullable` is transparent on the wire: it serializes exactly like
    /// the inner `Option`, so payloads keep their `null`/value shape.
    #[test]
    fn serializes_like_the_inner_option() {
        assert_eq!(
            serde_json::to_value(Nullable(Some(7u32))).unwrap(),
            serde_json::json!(7)
        );
        assert_eq!(
            serde_json::to_value(Nullable::<u32>(None)).unwrap(),
            serde_json::json!(null)
        );
    }

    #[test]
    fn deserializes_from_value_and_null() {
        let some: Nullable<u32> = serde_json::from_str("7").unwrap();
        assert_eq!(some.0, Some(7));
        let none: Nullable<u32> = serde_json::from_str("null").unwrap();
        assert_eq!(none.0, None);
    }

    /// The schema MCP clients see must offer the inner type *or* null —
    /// that `anyOf` is the whole point of this wrapper.
    #[test]
    fn schema_is_anyof_inner_or_null() {
        let schema = Nullable::<u32>::json_schema(&mut SchemaGenerator::default());
        let any_of = schema
            .as_value()
            .get("anyOf")
            .and_then(serde_json::Value::as_array)
            .expect("anyOf branch");
        assert_eq!(any_of.len(), 2);
        assert_eq!(any_of[1], serde_json::json!({ "type": "null" }));
        // The first branch is the inner type's schema (integer for u32).
        assert_eq!(any_of[0].get("type"), Some(&serde_json::json!("integer")));
    }

    #[test]
    fn schema_name_tracks_the_inner_type() {
        assert_eq!(
            Nullable::<u32>::schema_name(),
            format!("Nullable_{}", u32::schema_name())
        );
    }

    #[test]
    fn deref_and_deref_mut_reach_the_inner_option() {
        let mut n = Nullable(Some(1u32));
        assert_eq!(*n, Some(1u32));
        *n = Some(2u32);
        assert_eq!(n.0, Some(2u32));
        n.take();
        assert_eq!(n.0, None);
    }

    #[test]
    fn default_is_none() {
        assert_eq!(Nullable::<u32>::default().0, None);
    }

    #[test]
    fn converts_to_and_from_option() {
        let n: Nullable<u32> = Nullable::from(Some(3u32));
        let o: Option<u32> = n.into();
        assert_eq!(o, Some(3u32));
    }

    /// `#[serde(default)]` on a missing `Nullable` field — the shape the
    /// MCP `SearchRequest` relies on for `mode`.
    #[test]
    fn missing_field_defaults_to_none() {
        #[derive(serde::Deserialize)]
        struct WithDefault {
            #[serde(default)]
            mode: Nullable<u32>,
        }
        let v: WithDefault = serde_json::from_str("{}").unwrap();
        assert_eq!(v.mode.0, None);
    }
}
