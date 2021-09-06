use glfw::{Action, Context, Key};
use std::path::PathBuf;

static mut SCROLL_FACTOR: f32 = 0.1;
static mut MOUSE_X_POS: f64 = 0.0;
static mut MOUSE_Y_POS: f64 = 0.0;
static mut _CAMERA: Camera = Camera {
    pos: CameraPosition::Absolute {
        position: glam::Vec3::Z,
        look_at: glam::Vec3::ZERO,
    },
    zfar: 1000.0,
    znear: 0.1,
};
static mut CAMERA: Camera = Camera {
    pos: CameraPosition::SphericalAbout {
        origin: glam::Vec3::ZERO,
        radius: 3.0,
        theta: 3.14 / 2.0,
        phi: 0.0,
    },
    zfar: 1000.0,
    znear: 0.1,
};
static mut IS_PANNING: bool = false;
static mut ORIGINAL_X: f64 = 0.0;
static mut ORIGINAL_Y: f64 = 0.0;

enum CameraPosition {
    SphericalAbout {
        origin: glam::Vec3,
        radius: f32,
        theta: f32,
        phi: f32,
    },
    Absolute {
        position: glam::Vec3,
        look_at: glam::Vec3,
    },
}

struct Camera {
    pos: CameraPosition,
    zfar: f32,
    znear: f32,
}

fn pos_from_theta_phi(theta: f32, phi: f32) -> glam::Vec3 {
    glam::Vec3::new(theta.cos() * phi.cos(), phi.sin(), theta.sin() * phi.cos())
}

impl Camera {
    pub fn get_view(&self) -> glam::Mat4 {
        match self.pos {
            CameraPosition::Absolute { position, look_at } => {
                glam::Mat4::look_at_rh(position, look_at, glam::Vec3::new(0.0, 1.0, 0.0))
            }
            CameraPosition::SphericalAbout {
                origin,
                radius,
                theta,
                phi,
            } => {
                let pos = radius * pos_from_theta_phi(theta, phi);
                let pos = origin + pos;
                glam::Mat4::look_at_rh(
                    pos,
                    -(pos - origin).normalize(),
                    glam::Vec3::new(0.0, 1.0, 0.0),
                )
            }
        }
    }
}

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 5));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));
    let (mut window, events) = glfw
        .create_window(1600, 1080, "Lygre: glTF Loader", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.make_current();
    window.set_resizable(true);
    window.set_cursor_pos_polling(true);
    window.set_mouse_button_polling(true);
    window.set_key_polling(true);
    window.set_size_polling(true);
    window.set_scroll_polling(true);

    gl::load_with(|s| window.get_proc_address(s));

    let program;
    unsafe {
        let vshader = gl::CreateShader(gl::VERTEX_SHADER);
        let vert_source = std::ffi::CString::new(VERTEX_SOURCE).unwrap();
        gl::ShaderSource(
            vshader,
            1,
            std::mem::transmute(&vert_source),
            std::ptr::null(),
        );
        gl::CompileShader(vshader);

        let mut vertex_compiled = 0i32;
        gl::GetShaderiv(vshader, gl::COMPILE_STATUS, &mut vertex_compiled);

        if vertex_compiled as u8 != gl::TRUE {
            let mut log_length = 0;
            let mut message: [gl::types::GLchar; 1024] = [0; 1024];
            gl::GetShaderInfoLog(vshader, 1024, &mut log_length, message.as_mut_ptr());
            panic!(
                "Vertex shader error: {:?}",
                std::ffi::CString::from_raw(message.as_mut_ptr())
            );
        }

        let fshader = gl::CreateShader(gl::FRAGMENT_SHADER);
        let frag_source = std::ffi::CString::new(FRAG_SOURCE).unwrap();
        gl::ShaderSource(
            fshader,
            1,
            std::mem::transmute(&frag_source),
            std::ptr::null(),
        );
        gl::CompileShader(fshader);

        let mut frag_compiled = 0i32;
        gl::GetShaderiv(fshader, gl::COMPILE_STATUS, &mut frag_compiled);

        if frag_compiled as u8 != gl::TRUE {
            let mut log_length = 0;
            let mut message: [gl::types::GLchar; 1024] = [0; 1024];
            gl::GetShaderInfoLog(fshader, 1024, &mut log_length, message.as_mut_ptr());
            println!(
                "Fragment shader error: {:?}",
                std::ffi::CString::from_raw(message.as_mut_ptr())
            );
        }

        program = gl::CreateProgram();
        gl::AttachShader(program, vshader);
        gl::AttachShader(program, fshader);
        gl::LinkProgram(program);

        let mut program_linked = 0i32;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut program_linked);
        if program_linked as u8 != gl::TRUE {
            let mut log_length = 0;
            let mut message: [gl::types::GLchar; 1024] = [0; 1024];
            gl::GetProgramInfoLog(fshader, 1024, &mut log_length, message.as_mut_ptr());
            println!(
                "Program link error: {:?}",
                std::ffi::CString::from_raw(message.as_mut_ptr())
            );
        }

        gl::DeleteShader(vshader);
        gl::DeleteShader(fshader);
    }

    let mut filepath = PathBuf::new();
    filepath.push("res/scifi_helmet/scene.gltf");
    // filepath.push("res/damaged_helmet/DamagedHelmet.gltf");
    let document = gltf::Gltf::open(&filepath).unwrap();

    let mut raw_buffers = Vec::new();
    for buffer in document.buffers() {
        let source = buffer.source();
        if let gltf::buffer::Source::Uri(filename) = source {
            println!("Attempting to retrieve binary file {}", filename);
            raw_buffers.push(std::fs::read(filepath.with_file_name(filename)).unwrap());
        } else {
            unimplemented!();
        }
    }
    // let raw_buffers = vec![std::fs::read("res/scifi_helmet/scene.bin").unwrap()];

    let mut buffers = Vec::new();
    for rb in raw_buffers {
        unsafe {
            let mut vbo = 0;

            gl::GenBuffers(1, &mut vbo);

            let num_bytes = rb.len() as isize;
            println!("Buffering num_bytes={}", num_bytes);

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                num_bytes,
                std::mem::transmute(rb[..].as_ptr()),
                gl::STATIC_DRAW,
            );

            buffers.push((vbo, rb));
        }
    }

    let mut primitives = Vec::new();
    println!("Number of nodes: {}", document.nodes().len());

    let default_scene = document
        .default_scene()
        .unwrap_or(document.scenes().next().unwrap());

    let get_node_matrix = |node: &gltf::Node| match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => glam::Mat4::from_cols_array_2d(&matrix),
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::from_slice(&scale[..]),
            glam::Quat::from_array(rotation),
            glam::Vec3::from_slice(&translation[..]),
        ),
    };

    let mut levels = vec![default_scene
        .nodes()
        .map(|node| {
            let node_matrix = get_node_matrix(&node);
            (node, node_matrix)
        })
        .collect::<Vec<_>>()];

    print!("Root level: ");
    for (node, _) in levels[0].iter() {
        print!("Node#{} ", node.index());
    }
    println!();

    loop {
        let curr = &levels[levels.len() - 1];
        let mut to_append = Vec::new();
        for (node, node_matrix) in curr {
            for child in node.children() {
                let child_matrix = get_node_matrix(&child);
                // println!("Child has node matrix {:?}", child_matrix * *node_matrix);
                to_append.push((child, *node_matrix * child_matrix));
            }
        }

        print!("Next level: ");
        for (node, _) in to_append.iter() {
            print!("Node#{} ", node.index());
        }
        println!();

        if to_append.len() == 0 {
            break;
        }

        levels.push(to_append);
    }

    let all_nodes = levels.into_iter().flatten().collect::<Vec<_>>();

    for (node, node_matrix) in all_nodes {
        println!(
            "Node #{} has {} children",
            node.index(),
            node.children().count()
        );

        if let Some(cam_data) = node.camera() {
            if cam_data.index() != 0 {
                println!("Skipping camera#{}", cam_data.index());
            }
            match cam_data.projection() {
                gltf::camera::Projection::Orthographic(_) => {
                    unimplemented!();
                }
                gltf::camera::Projection::Perspective(p) => {
                    let (_scale, _rot, translation) =
                        get_node_matrix(&node).to_scale_rotation_translation();
                    println!("Setting camera radius to {}", translation.length());
                    unsafe {
                        CAMERA = Camera {
                            pos: CameraPosition::SphericalAbout {
                                origin: glam::Vec3::ZERO,
                                radius: translation.length(),
                                theta: 3.14 / 2.0,
                                phi: 0.0,
                            },
                            zfar: p.zfar().unwrap(),
                            znear: p.znear(),
                        };
                        SCROLL_FACTOR *= translation.length() / 3.0;
                    }
                }
            }
        }

        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                if gltf::mesh::Mode::Triangles != prim.mode() {
                    todo!("We only handle triangle meshes for now.");
                }
                unsafe {
                    let mut vao = 0;
                    gl::GenVertexArrays(1, &mut vao);

                    let vertex_attrib = |(attrib_idx, sem)| {
                        let accessor = prim.get(sem).unwrap();
                        let vbo = buffers[accessor.view().unwrap().buffer().index()].0;
                        gl::BindVertexArray(vao);
                        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
                        let attribute_size_bytes = accessor.size();
                        let attribute_multipicity = accessor.dimensions().multiplicity();
                        let stride = accessor
                            .view()
                            .unwrap()
                            .stride()
                            .unwrap_or(attribute_size_bytes);
                        let offset = accessor.offset() + accessor.view().unwrap().offset();

                        println!(
                            "Got accessor {{ size = {}, dimesions = {:?}, multiplicity = {}, normalized = {}, type = {:?} }}",
                            accessor.size(),
                            accessor.dimensions(),
                            accessor.dimensions().multiplicity(),
                            accessor.normalized(),
                            accessor.data_type()
                        );

                        // println!(
                        //     "Setting vertex_attrib {} for vbo={} as size={}, stride={}, offset={}",
                        //     attrib_idx, vbo, size, stride, offset
                        // );
                        gl::VertexAttribPointer(
                            attrib_idx,
                            attribute_multipicity as i32,
                            match accessor.data_type() {
                                gltf::accessor::DataType::F32 => gl::FLOAT,
                                other => {
                                    panic!("Wrong type for vertex attribute component: {:?}", other)
                                }
                            },
                            accessor.normalized() as u8,
                            stride as i32,
                            // .unwrap_or(size * std::mem::size_of::<f32>().try_into().unwrap()),
                            offset as *const std::ffi::c_void,
                        );
                        gl::EnableVertexAttribArray(attrib_idx);
                        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
                        gl::BindVertexArray(0);
                    };
                    vertex_attrib((0, &gltf::Semantic::Positions));
                    vertex_attrib((1, &gltf::Semantic::Normals));

                    let accessor = prim.indices().unwrap();
                    let indices = &buffers[accessor.view().unwrap().buffer().index()].1;
                    let indices_offset = accessor.offset() + accessor.view().unwrap().offset();
                    // println!(
                    //     "accessor.size()={}, accessor.count()={}, indices_offset={}",
                    //     accessor.size(),
                    //     accessor.count(),
                    //     indices_offset,
                    // );

                    println!(
                        "Buffering {} indices in total",
                        accessor.count() * accessor.size()
                    );

                    let mut ebo = 0;
                    gl::GenBuffers(1, &mut ebo);
                    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
                    gl::BufferData(
                        gl::ELEMENT_ARRAY_BUFFER,
                        accessor.count() as isize * accessor.size() as isize,
                        std::mem::transmute(&indices[indices_offset]),
                        gl::STATIC_DRAW,
                    );
                    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);

                    let material = prim.material();
                    let base_color: glam::Vec4 =
                        material.pbr_metallic_roughness().base_color_factor().into();

                    let base_color_tex_id = material
                        .pbr_metallic_roughness()
                        .base_color_texture()
                        .map(|info| {
                            let mut tex_id = 0u32;
                            gl::GenTextures(1, &mut tex_id);
                            gl::BindTexture(gl::TEXTURE_2D, tex_id);

                            let texture = info.texture();

                            // Set up texture sampling parameters.
                            let sampler = texture.sampler();
                            if let Some(mag_filter) = sampler.mag_filter() {
                                gl::TexParameteri(
                                    gl::TEXTURE_2D,
                                    gl::TEXTURE_MAG_FILTER,
                                    mag_filter.as_gl_enum() as i32,
                                );
                            }
                            if let Some(min_filter) = sampler.min_filter() {
                                gl::TexParameteri(
                                    gl::TEXTURE_2D,
                                    gl::TEXTURE_MIN_FILTER,
                                    min_filter.as_gl_enum() as i32,
                                );
                            }
                            gl::TexParameteri(
                                gl::TEXTURE_2D,
                                gl::TEXTURE_WRAP_S,
                                sampler.wrap_s().as_gl_enum() as i32,
                            );
                            gl::TexParameteri(
                                gl::TEXTURE_2D,
                                gl::TEXTURE_WRAP_T,
                                sampler.wrap_t().as_gl_enum() as i32,
                            );

                            let fmt_from_depth = |depth, is_u8| {
                                if depth == 1 {
                                    (if is_u8 { gl::R8 } else { gl::R32F }, gl::RED)
                                } else if depth == 2 {
                                    (if is_u8 { gl::RG8 } else { gl::RG32F }, gl::RG)
                                } else if depth == 3 {
                                    (if is_u8 { gl::RGB8 } else { gl::RGB32F }, gl::RGB)
                                } else if depth == 4 {
                                    (if is_u8 { gl::RGBA8 } else { gl::RGBA32F }, gl::RGBA)
                                } else {
                                    panic!("Invalid depth for 8-bit image {}", depth)
                                }
                            };

                            let image = texture.source();
                            match image.source() {
                                gltf::image::Source::Uri { uri, .. } => {
                                    match stb_image::image::load(filepath.with_file_name(uri)) {
                                        stb_image::image::LoadResult::Error(s) => panic!("{}", s),
                                        stb_image::image::LoadResult::ImageU8(img) => {
                                            println!(
                                                "Loaded an 8-bit image with {} channels named {}",
                                                img.depth, uri
                                            );
                                            gl::TexImage2D(
                                                gl::TEXTURE_2D,
                                                0,
                                                fmt_from_depth(img.depth, true).0 as i32,
                                                img.width as i32,
                                                img.height as i32,
                                                0,
                                                fmt_from_depth(img.depth, true).1,
                                                gl::UNSIGNED_BYTE,
                                                img.data.as_ptr() as *const std::ffi::c_void,
                                            );
                                        }
                                        stb_image::image::LoadResult::ImageF32(img) => {
                                            println!(
                                                "Loaded an 32-bit image with {} channels named {}",
                                                img.depth, uri
                                            );
                                            gl::TexImage2D(
                                                gl::TEXTURE_2D,
                                                0,
                                                fmt_from_depth(img.depth, false).0 as i32,
                                                img.width as i32,
                                                img.height as i32,
                                                0,
                                                fmt_from_depth(img.depth, false).1,
                                                gl::FLOAT,
                                                img.data.as_ptr() as *const std::ffi::c_void,
                                            );
                                        }
                                    }
                                }
                                gltf::image::Source::View { .. } => {
                                    unimplemented!();
                                }
                            }
                            gl::GenerateMipmap(gl::TEXTURE_2D);
                            gl::BindTexture(gl::TEXTURE_2D, 0);

                            tex_id
                        });

                    primitives.push((
                        vao,
                        ebo,
                        match accessor.data_type() {
                            gltf::accessor::DataType::U8 => gl::UNSIGNED_BYTE,
                            gltf::accessor::DataType::U16 => gl::UNSIGNED_SHORT,
                            gltf::accessor::DataType::U32 => gl::UNSIGNED_INT,
                            other => panic!("Invalid data type {:?} for indices", other),
                        },
                        0,
                        base_color_tex_id,
                        accessor.count() as i32,
                        base_color,
                        node_matrix,
                    ));
                }
            }
        }
    }

    unsafe {
        gl::Disable(gl::CULL_FACE);
        gl::Enable(gl::DEPTH_TEST);
    }

    let (width, height) = window.get_framebuffer_size();
    println!("{}, {}", width, height);
    unsafe {
        gl::Viewport(0, 0, width, height);
        gl::DebugMessageCallback(Some(gl_debug_callback), std::ptr::null());
        gl::DebugMessageControl(
            gl::DONT_CARE,
            gl::DONT_CARE,
            gl::DONT_CARE,
            0,
            std::ptr::null(),
            gl::TRUE,
        );
    }

    unsafe {
        gl::ClearColor(0.1, 0.1, 0.1, 1.0);
    }

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        unsafe {
            let view_matrix = CAMERA.get_view();
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            let (width, height) = window.get_framebuffer_size();
            let aspect_ratio = width as f32 / height as f32;
            let proj_matrix = glam::Mat4::perspective_rh_gl(
                90.0f32.to_radians() / aspect_ratio,
                aspect_ratio,
                CAMERA.znear,
                CAMERA.zfar,
            );

            let view_name = std::ffi::CString::new("u_view").unwrap();
            let loc = gl::GetUniformLocation(program, view_name.as_ptr() as *const i8);
            gl::ProgramUniformMatrix4fv(
                program,
                loc,
                1,
                gl::FALSE,
                view_matrix.to_cols_array().as_ptr(),
            );

            let proj_name = std::ffi::CString::new("u_proj").unwrap();
            let other_loc = gl::GetUniformLocation(program, proj_name.as_ptr() as *const i8);
            gl::ProgramUniformMatrix4fv(
                program,
                other_loc,
                1,
                gl::FALSE,
                proj_matrix.to_cols_array().as_ptr(),
            );

            for (
                vao,
                ebo,
                ebo_type,
                indices_offset,
                tex_id,
                num_indices,
                base_color,
                node_matrix,
            ) in primitives.iter()
            {
                let color_name = std::ffi::CString::new("u_object_color").unwrap();
                let color_loc = gl::GetUniformLocation(program, color_name.as_ptr() as *const i8);
                gl::ProgramUniform4fv(program, color_loc, 1, base_color.to_array().as_ptr());

                let model_name = std::ffi::CString::new("u_model").unwrap();
                let model_loc = gl::GetUniformLocation(program, model_name.as_ptr() as *const i8);
                gl::ProgramUniformMatrix4fv(
                    program,
                    model_loc,
                    1,
                    gl::FALSE,
                    node_matrix.to_cols_array().as_ptr(),
                );

                gl::UseProgram(program);

                // Note: Need to bind the VAO before the EBO, since the EBO will just point to the
                // previous VAO otherwise.
                gl::BindVertexArray(*vao);
                gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, *ebo);

                gl::DrawElements(
                    gl::TRIANGLES,
                    *num_indices,
                    *ebo_type,
                    *indices_offset as *const std::ffi::c_void,
                );
            }
        }

        window.swap_buffers();
    }
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        glfw::WindowEvent::Size(new_x, new_y) => {
            // window.set_size(new_x, new_y);
            println!("Resized to {} and {}", new_x, new_y);
            unsafe {
                gl::Viewport(0, 0, new_x, new_y);
            }
        }
        glfw::WindowEvent::MouseButton(glfw::MouseButton::Button3, glfw::Action::Press, _) => unsafe {
            IS_PANNING = true;
        },
        glfw::WindowEvent::MouseButton(glfw::MouseButton::Button3, glfw::Action::Release, _) => unsafe {
            IS_PANNING = false;
        },
        glfw::WindowEvent::Scroll(_, amount) => unsafe {
            match CAMERA.pos {
                CameraPosition::SphericalAbout {
                    origin: _,
                    ref mut radius,
                    theta: _,
                    phi: _,
                } => {
                    *radius -= (amount as f32) * SCROLL_FACTOR;
                }
                _ => {}
            }
        },
        glfw::WindowEvent::CursorPos(x, y) => unsafe {
            if IS_PANNING {
                match CAMERA.pos {
                    CameraPosition::Absolute {
                        ref mut position,
                        ref mut look_at,
                    } => {
                        *position += glam::Vec3::new(
                            -(x - MOUSE_X_POS) as f32,
                            (y - MOUSE_Y_POS) as f32,
                            0.0,
                        ) * 0.002;
                        *look_at += glam::Vec3::new(
                            -(x - MOUSE_X_POS) as f32,
                            (y - MOUSE_Y_POS) as f32,
                            0.0,
                        ) * 0.002;
                    }
                    CameraPosition::SphericalAbout {
                        origin: ref mut _origin,
                        radius: ref mut _radius,
                        ref mut phi,
                        ref mut theta,
                    } => {
                        // println!("theta={}, phi={}", theta.to_degrees(), phi.to_degrees());
                        if (x - ORIGINAL_X).abs() > (y - ORIGINAL_Y).abs() {
                            *theta += (x - MOUSE_X_POS) as f32 * 0.002;
                        } else {
                            *phi += (y - MOUSE_Y_POS) as f32 * 0.008;
                            *phi = phi.clamp(-3.14 / 2.0, 3.14 / 2.0);
                        }
                    }
                }
            } else {
                ORIGINAL_X = x;
                ORIGINAL_Y = y;
            }
            MOUSE_X_POS = x;
            MOUSE_Y_POS = y;
        },
        _ => {}
    }
}

const VERTEX_SOURCE: &'static str = "
#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 io_position;
out vec3 io_light_pos;
out vec3 io_normal;

void main() {
  io_light_pos = vec3(u_view * vec4(0.0, 2.0, 1.0, 1.0));
  io_position = vec3(u_view * u_model * vec4(a_pos, 1.0));
  io_normal = vec3(u_view * u_model * vec4(a_normal, 0.0));

  gl_Position = u_proj * u_view * u_model * vec4(a_pos, 1.0);
}
";

const FRAG_SOURCE: &'static str = "
#version 330 core

out vec4 FragColor;

vec4 k_light_color = vec4(1.0, 1.0, 1.0, 1.0);
// vec4 k_object_color = vec4(0.8, 0.2, 0.2, 1.0);
uniform vec4 u_object_color;

float k_ambient_coefficient = 0.4;
float k_diffuse_coefficient = 0.3;
float k_specular_coefficient = 0.3;
float k_p = 16;

in vec3 io_position;
in vec3 io_light_pos;
in vec3 io_normal;

void main() {

    vec3 to_light = normalize(io_light_pos - io_position);
    vec3 to_camera = normalize(-io_position);
    vec3 halfway = normalize(to_camera + to_light);
    float dist2 = dot(to_light, to_light);

    vec4 ambient_component = k_ambient_coefficient * u_object_color;

    vec4 diffuse_component = k_diffuse_coefficient * (k_light_color / dist2) * max(0, dot(io_normal, to_light));

    vec4 specular_component = k_specular_coefficient * (k_light_color / dist2) * pow(max(0, dot(io_normal, halfway)), k_p);

    FragColor = ambient_component + diffuse_component + specular_component;
}
";

extern "system" fn gl_debug_callback(
    _source: u32,
    _type: u32,
    _id: u32,
    _sev: u32,
    _length: i32,
    msg: *const i8,
    _data: *mut std::ffi::c_void,
) -> () {
    println!("OpenGL errored: {:?}", unsafe {
        std::ffi::CString::from_raw(std::mem::transmute(msg))
    });
}
