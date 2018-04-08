

/// Codec behaviour, used to store data.
pub trait Codec {
    type OutputType;
    fn encode(&self) -> String;
    fn decode(data: &str) -> Self::OutputType;
}