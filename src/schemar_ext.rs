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
