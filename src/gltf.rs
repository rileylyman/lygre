use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Gltf {
    pub asset: Asset,
    pub scene: usize,
    pub scenes: Vec<Scene>,
    pub nodes: Vec<Node>,
    pub meshes: Vec<Mesh>,
    pub accessors: Vec<Accessor>,
    pub materials: Vec<Material>,
    pub bufferViews: Vec<BufferView>,
    pub buffers: Vec<Buffer>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Asset {
    pub generator: String,
    pub version: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Primitive>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Scene {
    pub nodes: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Node {
    pub children: Option<Vec<usize>>,
    pub matrix: Option<Vec<f32>>,
    pub mesh: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Primitive {
    pub attributes: std::collections::HashMap<String, usize>,
    pub indices: usize,
    pub mode: usize,
    pub material: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Accessor {
    pub bufferView: usize,
    pub byteOffset: usize,
    pub componentType: usize,
    pub count: usize,
    pub max: serde_json::Value,
    pub min: serde_json::Value,
    pub r#type: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Material {
    pub pbrMetallicRoughness: PbrMetallicRoughness,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PbrMetallicRoughness {
    pub baseColorFactor: [f32; 4],
    pub metallicFactor: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BufferView {
    pub buffer: usize,
    pub byteOffset: usize,
    pub byteLength: usize,
    pub byteStride: Option<usize>,
    pub target: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Buffer {
    pub byteLength: usize,
    pub uri: String,
}
