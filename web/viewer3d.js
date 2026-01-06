// Local Three.js bundle paths (no CDN latency)
const THREE_URL = "./lib/three/three.module.js";
const FBX_LOADER_URL = "./lib/three/examples/jsm/loaders/FBXLoader.js";
const ORBIT_CONTROLS_URL = "./lib/three/examples/jsm/controls/OrbitControls.js";
const TRANSFORM_CONTROLS_URL = "./lib/three/examples/jsm/controls/TransformControls.js";
const GLTF_EXPORTER_URL = "./lib/three/examples/jsm/exporters/GLTFExporter.js";
// Note: GLTFLoader and OBJLoader will be loaded from CDN as fallback if not bundled
const GLTF_LOADER_URL = "https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";
const OBJ_LOADER_URL = "https://esm.sh/three@0.160.0/examples/jsm/loaders/OBJLoader.js";

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";


console.log("[HY-Motion] app imported successfully:", !!app);

app.registerExtension({
    name: "HYMotion.3DViewer",
    async init() {
        console.log("[HY-Motion] 3D Viewer Extension Loaded (v3.4 - Asset Bake Fix)");
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "HYMotion3DViewer" &&
            nodeData.name !== "HYMotionFBXPlayer" &&
            nodeData.name !== "HYMotion3DModelLoader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            console.log("[HY-Motion] Creating 3D Viewer Node:", node.id);

            // Main container with dynamic height
            const defaultHeight = 400;
            const storedHeight = localStorage.getItem('hymotion_viewer_height') || defaultHeight;

            const container = document.createElement("div");
            container.style.cssText = `width:100%; height:100%; background:#111; position:relative; display:flex; flex-direction:column; border:1px solid #333; border-radius:4px; overflow:hidden;`;

            const canvasContainer = document.createElement("div");
            canvasContainer.style.cssText = "flex:1; width:100%; height:100%; min-height:0; overflow:hidden; position:relative;";
            container.appendChild(canvasContainer);

            // Add custom slider styles
            const style = document.createElement("style");
            style.textContent = `
                .hymotion-slider {
                    -webkit-appearance: none;
                    appearance: none;
                    background: transparent;
                    cursor: pointer;
                }
                .hymotion-slider::-webkit-slider-runnable-track {
                    background: #444;
                    height: 8px;
                    border-radius: 4px;
                }
                .hymotion-slider::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    margin-top: -6px; /* Centers thumb on the 8px track */
                    background-color: #007bff;
                    height: 20px;
                    width: 20px;
                    border-radius: 50%;
                    border: 2px solid #fff;
                    box-shadow: 0 0 5px rgba(0,0,0,0.5);
                    transition: transform 0.1s ease;
                }
                .hymotion-slider::-webkit-slider-thumb:hover {
                    transform: scale(1.2);
                    background-color: #008cff;
                }
                .hymotion-slider::-moz-range-track {
                    background: #444;
                    height: 8px;
                    border-radius: 4px;
                }
                .hymotion-slider::-moz-range-thumb {
                    background-color: #007bff;
                    height: 18px;
                    width: 18px;
                    border-radius: 50%;
                    border: 2px solid #fff;
                    box-shadow: 0 0 5px rgba(0,0,0,0.5);
                }
            `;
            container.appendChild(style);

            // Playback controls
            const controls = document.createElement("div");
            controls.style.cssText = "width:100%; box-sizing:border-box; height:auto; min-height:50px; display:flex; flex-wrap:wrap; align-items:center; padding:8px 10px; gap:6px; background:#222; overflow:hidden; flex-shrink:0;";


            const playBtn = document.createElement("button");
            playBtn.innerText = "Play";
            playBtn.style.cssText = "cursor:pointer; padding:6px 16px; font-size:14px; font-weight:600; flex-shrink:0; white-space:nowrap;";

            const progress = document.createElement("input");
            progress.type = "range";
            progress.className = "hymotion-slider";
            progress.min = 0;
            progress.max = 100;
            progress.step = "any";
            progress.value = 0;
            progress.style.flex = "1 1 auto";
            progress.style.minWidth = "80px";
            progress.style.margin = "0 10px";

            let isScrubbing = false;
            progress.onmousedown = progress.ontouchstart = () => { isScrubbing = true; };
            const releaseScrubbing = () => { isScrubbing = false; };
            window.addEventListener('mouseup', releaseScrubbing, { passive: true });
            window.addEventListener('touchend', releaseScrubbing, { passive: true });
            progress.onchange = releaseScrubbing; // Fallback

            const statusLabel = document.createElement("div");
            statusLabel.style.cssText = "font-size:13px; color:#888; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:180px;";
            statusLabel.innerText = "Ready";

            const cycleBtn = document.createElement("button");
            cycleBtn.innerText = "Select";
            cycleBtn.title = "Cycle through loaded models and skeletons";
            cycleBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#444; color:#fff; border:1px solid #666; border-radius:3px; font-weight:500; flex-shrink:0; white-space:nowrap;";

            const exportBtn = document.createElement("button");
            exportBtn.innerText = "Export";
            exportBtn.title = "Export or Download selection";
            exportBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#226622; color:#fff; border:1px solid #338833; border-radius:3px; display:none; font-weight:500; flex-shrink:0; white-space:nowrap;";

            // Gizmo mode buttons (on left side)
            const gizmoGroup = document.createElement("div");
            gizmoGroup.style.cssText = "display:flex; gap:2px; border-right:1px solid #444; padding-right:6px; margin-right:6px; flex-shrink:0;";

            const createGizmoBtn = (mode, icon, tooltip) => {
                const btn = document.createElement("button");
                btn.innerText = icon;
                btn.title = tooltip;
                btn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:14px; background:#333; color:#aaa; border:1px solid #555; border-radius:3px; font-weight:bold; flex-shrink:0;";
                btn.dataset.mode = mode;
                return btn;
            };

            const translateBtn = createGizmoBtn('translate', '⬌', 'Translate (Move) - G key');
            const rotateBtn = createGizmoBtn('rotate', '↻', 'Rotate - R key');
            const scaleBtn = createGizmoBtn('scale', '⊡', 'Scale - S key');
            const gizmoOffBtn = createGizmoBtn('none', '✕', 'Disable Gizmo - Esc key');

            gizmoGroup.appendChild(translateBtn);
            gizmoGroup.appendChild(rotateBtn);
            gizmoGroup.appendChild(scaleBtn);
            gizmoGroup.appendChild(gizmoOffBtn);

            const applyBtn = document.createElement("button");
            applyBtn.innerText = "Apply";
            applyBtn.title = "Apply current transform widgets to the model's base pose and reset them";
            applyBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#2266aa; color:#fff; border:1px solid #3388ff; border-radius:3px; display:none; font-weight:500; flex-shrink:0; white-space:nowrap;";

            const transformBtn = document.createElement("button");
            transformBtn.innerText = "Transform";
            transformBtn.title = "Open transform panel";
            transformBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#444; color:#fff; border:1px solid #666; border-radius:3px; display:none; font-weight:500; flex-shrink:0; white-space:nowrap;";

            controls.appendChild(gizmoGroup);
            controls.appendChild(playBtn);
            controls.appendChild(cycleBtn);
            controls.appendChild(exportBtn);
            controls.appendChild(transformBtn);
            controls.appendChild(applyBtn);
            controls.appendChild(progress);
            controls.appendChild(statusLabel);
            container.appendChild(controls);

            // Transform Overlay Panel
            const transformPanel = document.createElement("div");
            transformPanel.style.cssText = "position:absolute; top:10px; right:10px; width:220px; background:rgba(20,20,20,0.85); backdrop-filter:blur(5px); border:1px solid #444; border-radius:8px; display:none; flex-direction:column; padding:12px; gap:10px; z-index:2000; color:#eee; font-family:sans-serif; user-select:none;";
            canvasContainer.appendChild(transformPanel);

            const refreshTransforms = () => {
                try {
                    if (!currentModel) return;
                    const obj = loadedModels.find(m => m.model === currentModel);
                    if (!obj) {
                        console.warn("[HY-Motion] refreshTransforms: model not found in loadedModels");
                        return;
                    }

                    const getVal = (name, def) => {
                        const w = node.widgets?.find(widget => widget.name === name);
                        return w ? parseFloat(w.value) || 0 : def;
                    };

                    const tx = getVal("translate_x", 0);
                    const ty = getVal("translate_y", 0);
                    const tz = getVal("translate_z", 0);
                    const rx = getVal("rotate_x", 0);
                    const ry = getVal("rotate_y", 0);
                    const rz = getVal("rotate_z", 0);
                    const sx = getVal("scale_x", 1);
                    const sy = getVal("scale_y", 1);
                    const sz = getVal("scale_z", 1);

                    currentModel.position.set(obj.basePosition.x + tx, obj.basePosition.y + ty, obj.basePosition.z + tz);
                    currentModel.rotation.set(
                        (obj.baseRotation.x + rx) * Math.PI / 180,
                        (obj.baseRotation.y + ry) * Math.PI / 180,
                        (obj.baseRotation.z + rz) * Math.PI / 180
                    );
                    currentModel.scale.set(obj.baseScale.x * sx, obj.baseScale.y * sy, obj.baseScale.z * sz);

                    // Sync panel inputs (except the one the user is currently typing in)
                    const pInputs = transformPanel.querySelectorAll('input');
                    pInputs.forEach(input => {
                        const key = input.dataset.key;
                        if (document.activeElement === input) return;
                        if (key.startsWith("translate_")) {
                            const axis = key.split("_")[1];
                            const val = axis === "x" ? tx : (axis === "y" ? ty : tz);
                            input.value = val.toFixed(2);
                        } else if (key.startsWith("rotate_")) {
                            const axis = key.split("_")[1];
                            const val = axis === "x" ? rx : (axis === "y" ? ry : rz);
                            input.value = val.toFixed(2);
                        } else if (key.startsWith("scale_")) {
                            const axis = key.split("_")[1];
                            const val = axis === "x" ? sx : (axis === "y" ? sy : sz);
                            input.value = val.toFixed(2);
                        }
                    });

                    requestRender();
                    if (app.graph) app.graph.setDirtyCanvas(true);
                } catch (e) {
                    console.error("[HY-Motion] refreshTransforms Error:", e);
                }
            };

            const syncWidgetsFromModel = (model) => {
                try {
                    if (!model) return;
                    const obj = loadedModels.find(m => m.model === model);
                    if (!obj) return;

                    const setWidget = (name, val) => {
                        const w = node.widgets?.find(widget => widget.name === name);
                        if (w) w.value = val;
                    };

                    // Calc relative position
                    setWidget("translate_x", model.position.x - obj.basePosition.x);
                    setWidget("translate_y", model.position.y - obj.basePosition.y);
                    setWidget("translate_z", model.position.z - obj.basePosition.z);

                    // Calc relative rotation (convert rad to deg)
                    setWidget("rotate_x", (model.rotation.x * 180 / Math.PI) - obj.baseRotation.x);
                    setWidget("rotate_y", (model.rotation.y * 180 / Math.PI) - obj.baseRotation.y);
                    setWidget("rotate_z", (model.rotation.z * 180 / Math.PI) - obj.baseRotation.z);

                    // Calc relative scale
                    setWidget("scale_x", model.scale.x / (obj.baseScale.x || 1));
                    setWidget("scale_y", model.scale.y / (obj.baseScale.y || 1));
                    setWidget("scale_z", model.scale.z / (obj.baseScale.z || 1));

                    // Update transform panel inputs
                    const pInputs = transformPanel.querySelectorAll('input');
                    pInputs.forEach(input => {
                        const key = input.dataset.key;
                        if (document.activeElement === input) return;
                        const w = node.widgets?.find(widget => widget.name === key);
                        if (w) input.value = parseFloat(w.value).toFixed(2);
                    });

                    if (app.graph) app.graph.setDirtyCanvas(true);
                } catch (e) {
                    console.error("[HY-Motion] syncWidgetsFromModel Error:", e);
                }
            };

            const createTransformRow = (label, prefix, defaultVal) => {
                const row = document.createElement("div");
                row.style.cssText = "display:flex; flex-direction:column; gap:4px;";
                const header = document.createElement("div");
                header.innerText = label;
                header.style.cssText = "font-size:11px; font-weight:bold; color:#aaa; text-transform:uppercase;";
                row.appendChild(header);

                const inputs = document.createElement("div");
                inputs.style.cssText = "display:flex; gap:4px;";
                ['x', 'y', 'z'].forEach(axis => {
                    const input = document.createElement("input");
                    input.type = "text";
                    input.value = defaultVal;
                    input.style.cssText = "width:100%; height:24px; background:#333; color:#fff; border:1px solid #555; border-radius:3px; font-size:12px; padding:2px 4px; outline:none;";
                    input.dataset.key = `${prefix}_${axis}`;

                    input.oninput = (e) => {
                        const val = parseFloat(input.value) || 0;
                        const w = node.widgets?.find(widget => widget.name === input.dataset.key);
                        if (w) {
                            w.value = val;
                        }
                        refreshTransforms();
                    };
                    inputs.appendChild(input);
                });
                row.appendChild(inputs);
                return row;
            };

            transformPanel.appendChild(createTransformRow("Position", "translate", 0));
            transformPanel.appendChild(createTransformRow("Rotation (Deg)", "rotate", 0));
            transformPanel.appendChild(createTransformRow("Scale", "scale", 1));

            transformBtn.onclick = (e) => {
                e.stopPropagation();
                transformPanel.style.display = transformPanel.style.display === "none" ? "flex" : "none";
                transformBtn.style.background = transformPanel.style.display === "none" ? "#444" : "#666";
            };

            // Hide playback controls for 3D Model Loader
            if (nodeData.name === "HYMotion3DModelLoader") {
                playBtn.style.display = "none";
                progress.style.display = "none";
            }

            // Hide gizmo controls for Legacy FBX Player
            if (nodeData.name === "HYMotionFBXPlayer") {
                gizmoGroup.style.display = "none";
            }

            // Resize handle at the bottom
            const resizeHandle = document.createElement("div");
            resizeHandle.style.cssText = "position:absolute; bottom:0; left:0; right:0; height:6px; background:linear-gradient(to bottom, transparent, #444); cursor:ns-resize; z-index:1000;";
            resizeHandle.title = "Drag to resize viewer height";

            let isResizing = false;
            let startY = 0;
            let startHeight = 0;

            resizeHandle.addEventListener('mousedown', (e) => {
                isResizing = true;
                startY = e.clientY;
                startHeight = node.size[1]; // Use node height instead of container
                e.preventDefault();
            });

            document.addEventListener('mousemove', (e) => {
                if (!isResizing) return;
                const delta = e.clientY - startY;
                const newHeight = Math.max(280, Math.min(1280, startHeight + delta)); // Min 280px (200 canvas + 80 controls), max 1280px

                // Update node size to respect viewer height
                node.size[1] = newHeight;
                localStorage.setItem('hymotion_viewer_height', newHeight - 90);

                // Force immediate renderer resize
                requestAnimationFrame(() => {
                    if (renderer && camera && canvasContainer.clientWidth > 0 && canvasContainer.clientHeight > 0) {
                        camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight, false);
                    }
                });
            });

            document.addEventListener('mouseup', () => {
                if (isResizing) {
                    isResizing = false;

                    // Save final node size
                    localStorage.setItem('hymotion_viewer_height', node.size[1] - 90);

                    // Final sync after resize complete
                    setTimeout(() => {
                        if (renderer && camera && canvasContainer.clientWidth > 0 && canvasContainer.clientHeight > 0) {
                            camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
                            camera.updateProjectionMatrix();
                            renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
                        }
                    }, 50);
                }
            });

            container.appendChild(resizeHandle);

            this.addDOMWidget("3d_viewer", "viewer", container);
            this.size = [400, parseInt(storedHeight) + 90]; // Add some padding for controls

            let THREE = window.__HY_MOTION_THREE__ || null;
            let renderer, scene, camera, orbitControls;
            let currentModel = null, mixer = null;
            let skeletalSamples = [];
            let isPlaying = false;
            let clock = null;
            let currentFrame = 0;
            let maxFrames = 0;
            let frameAccumulator = 0;
            const targetFPS = 30;
            const frameTime = 1 / targetFPS;
            let mixers = [];
            let isInitialized = false;
            let isInitializing = false;
            let pendingDataQueue = [];
            let lastMotionsData = null;
            let lastFbxUrl = null;
            let lastModelUrl = null;
            let animationFrameId = null;

            // Performance optimization flags
            let needsRender = true; // Dirty flag for smart rendering
            let isAnimating = false; // Track if animation loop should run

            let raycaster = null;
            let selectedModels = []; // Array to support multi-selection
            let loadedModels = [];

            const SMPL_H_SKELETON = [
                [0, 1], [0, 2], [0, 3],
                [1, 4], [2, 5], [3, 6],
                [4, 7], [5, 8], [6, 9],
                [7, 10], [8, 11], [9, 12],
                [9, 13], [9, 14], // Shoulders connect to Spine3 (9), not Neck (12)
                [12, 15], // Neck to Head
                [13, 16], [14, 17],
                [16, 18], [17, 19],
                [18, 20], [19, 21],
                // Left Fingers
                [20, 22], [22, 23], [23, 24], // Index
                [20, 25], [25, 26], [26, 27], // Middle
                [20, 28], [28, 29], [29, 30], // Pinky
                [20, 31], [31, 32], [32, 33], // Ring
                [20, 34], [34, 35], [35, 36], // Thumb
                // Right Fingers
                [21, 37], [37, 38], [38, 39], // Index
                [21, 40], [40, 41], [41, 42], // Middle
                [21, 43], [43, 44], [44, 45], // Pinky
                [21, 46], [46, 47], [47, 48], // Ring
                [21, 49], [49, 50], [50, 51]  // Thumb
            ];

            const initThree = async () => {
                if (isInitialized || isInitializing) return;
                isInitializing = true;
                try {
                    // Load all libraries in parallel to reduce freeze time
                    if (!window.__HY_MOTION_THREE__) {
                        console.log("[HY-Motion] Loading Three.js libs...");
                        const [three, orbit, exporter] = await Promise.all([
                            import(THREE_URL),
                            import(ORBIT_CONTROLS_URL),
                            import(GLTF_EXPORTER_URL)
                        ]);
                        window.__HY_MOTION_THREE__ = three;
                        window.__HY_MOTION_ORBIT__ = orbit;
                        window.__HY_MOTION_EXPORTER__ = exporter;
                    }
                    THREE = window.__HY_MOTION_THREE__;
                    const { OrbitControls } = window.__HY_MOTION_ORBIT__;
                    const { GLTFExporter } = window.__HY_MOTION_EXPORTER__;

                    if (!scene) {
                        scene = new THREE.Scene();
                        scene.background = new THREE.Color(0x111111);

                        camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
                        camera.position.set(0, 2, 5);

                        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

                        const width = canvasContainer.clientWidth || 400;
                        const height = canvasContainer.clientHeight || 400;
                        renderer.setSize(width, height, false); // Use false to avoid updating CSS size directly
                        camera.aspect = width / height;
                        camera.updateProjectionMatrix();

                        // Force canvas to fill container via CSS
                        renderer.domElement.style.width = '100%';
                        renderer.domElement.style.height = '100%';
                        renderer.domElement.style.display = 'block';

                        // Store references on canvas for onResize callback access
                        renderer.domElement.__renderer = renderer;
                        renderer.domElement.__camera = camera;

                        canvasContainer.appendChild(renderer.domElement);

                        orbitControls = new OrbitControls(camera, renderer.domElement);
                        orbitControls.enableDamping = true;
                        orbitControls.dampingFactor = 0.05;
                        orbitControls.target.set(0, 0.8, 0);

                        // Trigger rendering when user interacts with camera
                        orbitControls.addEventListener('start', () => {
                            startAnimating(); // Start loop when user begins interaction
                        });
                        orbitControls.addEventListener('change', () => {
                            requestRender(); // Request render on camera change
                        });

                        // Allow middle mouse to rotate (like Blender)
                        orbitControls.mouseButtons = {
                            LEFT: THREE.MOUSE.ROTATE,
                            MIDDLE: THREE.MOUSE.ROTATE,
                            RIGHT: THREE.MOUSE.PAN
                        };

                        // Initialize TransformControls (gizmos)
                        if (!window.__HY_MOTION_TRANSFORM__) {
                            window.__HY_MOTION_TRANSFORM__ = await import(TRANSFORM_CONTROLS_URL);
                        }
                        const { TransformControls } = window.__HY_MOTION_TRANSFORM__;
                        const transformControl = new TransformControls(camera, renderer.domElement);
                        scene.add(transformControl);

                        // Prevent orbit controls from interfering with gizmo
                        transformControl.addEventListener('dragging-changed', (event) => {
                            orbitControls.enabled = !event.value;
                            if (event.value) {
                                startAnimating(); // Start loop when dragging gizmo
                            }
                        });
                        transformControl.addEventListener('change', () => {
                            if (transformControl.object && currentModel === transformControl.object) {
                                syncWidgetsFromModel(currentModel);
                            }
                            requestRender();
                        });

                        // Gizmo mode switching function
                        let currentGizmoMode = 'none';
                        const setGizmoMode = (mode) => {
                            currentGizmoMode = mode;

                            // Update button styles
                            [translateBtn, rotateBtn, scaleBtn, gizmoOffBtn].forEach(btn => {
                                if (btn.dataset.mode === mode) {
                                    btn.style.background = '#0066cc';
                                    btn.style.color = '#fff';
                                } else {
                                    btn.style.background = '#333';
                                    btn.style.color = '#aaa';
                                }
                            });

                            // Set gizmo mode
                            if (mode === 'none') {
                                transformControl.detach();
                            } else {
                                transformControl.setMode(mode);
                                // Attach to first selected model if any
                                if (selectedModels.length > 0 && selectedModels[0].type === 'model') {
                                    transformControl.attach(selectedModels[0].obj.model);
                                }
                            }
                        };

                        // Button click handlers
                        translateBtn.onclick = () => setGizmoMode('translate');
                        rotateBtn.onclick = () => setGizmoMode('rotate');
                        scaleBtn.onclick = () => setGizmoMode('scale');
                        gizmoOffBtn.onclick = () => setGizmoMode('none');

                        // Keyboard shortcuts (Blender-style) - only when hovering over canvas
                        let isHoveringCanvas = false;

                        // Track mouse hover state
                        canvasContainer.addEventListener('mouseenter', () => {
                            isHoveringCanvas = true;
                        });
                        canvasContainer.addEventListener('mouseleave', () => {
                            isHoveringCanvas = false;
                        });

                        const handleKeyPress = (e) => {
                            // Only handle shortcuts when hovering over the canvas
                            if (!isHoveringCanvas) return;
                            // Disable gizmo shortcuts for Legacy Player
                            if (nodeData.name === "HYMotionFBXPlayer") return;
                            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                            switch (e.key.toLowerCase()) {
                                case 'g':
                                    setGizmoMode('translate');
                                    e.preventDefault();
                                    e.stopPropagation();
                                    break;
                                case 'r':
                                    setGizmoMode('rotate');
                                    e.preventDefault();
                                    e.stopPropagation();
                                    break;
                                case 's':
                                    setGizmoMode('scale');
                                    e.preventDefault();
                                    e.stopPropagation();
                                    break;
                                case 'escape':
                                    setGizmoMode('none');
                                    e.preventDefault();
                                    e.stopPropagation();
                                    break;
                            }
                        };

                        document.addEventListener('keydown', handleKeyPress);

                        // Store for later use
                        this.transformControl = transformControl;
                        this.setGizmoMode = setGizmoMode;

                        scene.add(new THREE.AmbientLight(0xffffff, 1.0));
                        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
                        dirLight.position.set(5, 10, 7.5);
                        scene.add(dirLight);

                        const grid = new THREE.GridHelper(10, 10, 0x666666, 0x333333);
                        scene.add(grid);

                        const originMarker = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.05), new THREE.MeshBasicMaterial({ color: 0xff0000 }));
                        scene.add(originMarker);

                        const groundGeo = new THREE.PlaneGeometry(20, 20);
                        const groundMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a });
                        const ground = new THREE.Mesh(groundGeo, groundMat);
                        ground.rotation.x = -Math.PI / 2;
                        ground.position.y = -0.01;
                        scene.add(ground);

                        clock = new THREE.Clock();

                        // Throttle resize events to prevent excessive updates
                        let resizeTimeout;
                        const resizeObserver = new ResizeObserver(() => {
                            clearTimeout(resizeTimeout);
                            resizeTimeout = setTimeout(() => {
                                if (canvasContainer.clientWidth > 0 && canvasContainer.clientHeight > 0) {
                                    camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
                                    camera.updateProjectionMatrix();
                                    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight, false);
                                    requestRender(); // Trigger single render after resize
                                }
                            }, 100); // Throttle to 100ms
                        });
                        resizeObserver.observe(canvasContainer);

                        // Initialize hit proxies and highlights whenever we are rendering to ensure consistency
                        raycaster = new THREE.Raycaster();

                        // Optimized raycasting with throttling
                        let lastRaycastTime = 0;
                        const raycastInterval = 100; // 10fps for hover detection

                        const throttledPointerMove = (event) => {
                            const now = Date.now();
                            if (now - lastRaycastTime < raycastInterval) return;
                            lastRaycastTime = now;
                            onCanvasPointerMove(event);
                        };

                        canvasContainer.addEventListener('pointerdown', onCanvasPointerDown);
                        canvasContainer.addEventListener('pointermove', throttledPointerMove);

                        // Intersection Observer to pause rendering when not visible
                        const visibilityObserver = new IntersectionObserver((entries) => {
                            entries.forEach(entry => {
                                if (entry.isIntersecting) {
                                    startAnimating();
                                } else {
                                    stopAnimating();
                                }
                            });
                        }, { threshold: 0.1 });
                        visibilityObserver.observe(container);

                        // Store observers for cleanup
                        this._visibilityObserver = visibilityObserver;
                        this._resizeObserver = resizeObserver;
                        this._handleKeyPress = handleKeyPress;
                    }

                    isInitialized = true;
                    isInitializing = false;
                    while (pendingDataQueue.length > 0) handleData(pendingDataQueue.shift());
                    if (animationFrameId) cancelAnimationFrame(animationFrameId);
                    startAnimating(); // Start with initial render
                } catch (e) {
                    console.error("[HY-Motion] Viewer Init Error:", e);
                    statusLabel.innerText = "Error: Three.js fail";
                }
            };

            const getIntersectables = () => {
                const intersectableObjects = [];
                for (const obj of loadedModels) {
                    if (obj.hitProxy) intersectableObjects.push(obj.hitProxy);
                }
                for (const sample of skeletalSamples) {
                    for (const joint of sample.joints) {
                        intersectableObjects.push(joint);
                    }
                }
                return intersectableObjects;
            };

            const onCanvasPointerMove = (event) => {
                if (!raycaster || !camera || !scene) return;
                const rect = renderer.domElement.getBoundingClientRect();
                const mouse = new THREE.Vector2();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);

                // Check intersects using proxies/joints
                const intersects = raycaster.intersectObjects(getIntersectables(), false);
                renderer.domElement.style.cursor = intersects.length > 0 ? "pointer" : "auto";
            };

            const onCanvasPointerDown = (event) => {
                if (!raycaster || !camera || !scene) return;

                // Prevent interfering with orbit controls mouse/touch
                if (event.button !== 0) return;

                const rect = renderer.domElement.getBoundingClientRect();
                const mouse = new THREE.Vector2();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(getIntersectables(), false);

                if (intersects.length > 0) {
                    // Find first valid hit
                    let hit = null;
                    for (let i = 0; i < intersects.length; i++) {
                        if (intersects[i].object.userData.selectableParent) {
                            hit = intersects[i].object;
                            break;
                        }
                    }

                    if (hit) {
                        selectObject(hit.userData.selectableParent, hit.userData.type, event.ctrlKey);
                        event.stopPropagation();
                        // DO NOT preventDefault here as it might break OrbitControls
                    }
                } else {
                    if (!event.ctrlKey) deselectAll();
                }
            };


            const cycleSelection = () => {
                const all = [
                    ...loadedModels.map(m => ({ obj: m, type: "model" })),
                    ...skeletalSamples.map(s => ({ obj: s, type: "skeleton" }))
                ];
                if (all.length === 0) return;

                let nextIdx = 0;
                if (selectedModels.length > 0) {
                    const lastSelected = selectedModels[selectedModels.length - 1];
                    const currentIdx = all.findIndex(item => item.obj === lastSelected.obj);
                    nextIdx = (currentIdx + 1) % all.length;
                }
                const next = all[nextIdx];
                deselectAll(); // Cycle clears others for clarity
                selectObject(next.obj, next.type, false);
            };

            const getRealtimeBox = (obj) => {
                const box = new THREE.Box3();
                obj.traverse((child) => {
                    if (child.isMesh) {
                        if (child.isSkinnedMesh && child.skeleton) {
                            // Find bounds of all bones in world space as an approximation of the volume
                            const boneBox = new THREE.Box3();
                            child.skeleton.bones.forEach(bone => {
                                boneBox.expandByPoint(bone.getWorldPosition(new THREE.Vector3()));
                            });
                            // Buff slightly to cover the mesh thickness
                            boneBox.expandByScalar(0.15);
                            box.union(boneBox);
                        } else {
                            const childBox = new THREE.Box3().setFromObject(child);
                            if (!childBox.isEmpty()) box.union(childBox);
                        }
                    }
                });
                return box;
            };

            const selectObject = (obj, type, isMulti = false) => {
                if (!isMulti) deselectAll();

                // Toggle if already selected
                const existingIdx = selectedModels.findIndex(m => m.obj === obj);
                if (existingIdx !== -1) {
                    if (isMulti) {
                        removeSelectionAt(existingIdx);
                        updateUI();
                        return;
                    } else {
                        // If clicking same without multi, just keep it (deselect others happened above)
                    }
                } else {
                    const selection = { obj, type };
                    applyHighlight(selection);
                    selectedModels.push(selection);
                }

                if (type === "model") {
                    currentModel = obj.model;
                    syncWidgetsFromModel(currentModel); // Sync UI TO this model's state
                }

                // Auto-attach gizmo if in gizmo mode and model is selected (except for Legacy Player)
                if (nodeData.name !== "HYMotionFBXPlayer" && this.transformControl && this.setGizmoMode && type === 'model') {
                    const currentMode = [translateBtn, rotateBtn, scaleBtn].find(btn =>
                        btn.style.background === 'rgb(0, 102, 204)' || btn.style.background === '#0066cc'
                    )?.dataset.mode;

                    if (currentMode) {
                        this.transformControl.attach(obj.model);
                    }
                }

                updateUI();
                updateSelectionHighlights(); // Update highlights immediately
                requestRender(); // Trigger render
            };

            const applyHighlight = (selection) => {
                const { obj, type } = selection;
                if (type === "model") {
                    const target = obj.model;
                    target.traverse((child) => {
                        if (child.isMesh) {
                            const mats = Array.isArray(child.material) ? child.material : [child.material];
                            for (const m of mats) {
                                if (!m) continue;
                                if (!m.userData) m.userData = {};
                                if (!m.userData.origEmissive) m.userData.origEmissive = m.emissive ? m.emissive.clone() : new THREE.Color(0, 0, 0);
                                m.emissive = new THREE.Color(0x00ff00);
                                m.emissiveIntensity = 0.6;
                            }
                        }
                    });

                    const box = new THREE.BoxHelper(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)), 0x00ff00);
                    box.name = "selection_highlight";
                    box.visible = false; // Hidden as requested: "dont remove only hide"
                    scene.add(box);
                    selection.highlight = box;

                } else if (type === "skeleton") {
                    for (const joint of obj.joints) {
                        joint.material.emissive = new THREE.Color(0xffff00);
                        joint.material.emissiveIntensity = 1.0;
                    }

                    const box = new THREE.BoxHelper(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)), 0xffff00);
                    box.name = "selection_highlight";
                    box.visible = false; // Hidden as requested: "dont remove only hide"
                    scene.add(box);
                    selection.highlight = box;
                }
            };

            const updateUI = () => {
                const count = selectedModels.length;
                const isLegacyPlayer = nodeData.name === "HYMotionFBXPlayer";

                if (count === 0) {
                    statusLabel.innerText = "Ready";
                    statusLabel.style.color = "#888";
                    exportBtn.style.display = "none";
                } else if (count === 1) {
                    const s = selectedModels[0];
                    statusLabel.innerText = `Selected ${s.type === 'model' ? 'Model' : 'Skeleton'}: ${s.obj.name || ''}`;
                    statusLabel.style.color = "#0f0";

                    if (isLegacyPlayer) {
                        exportBtn.style.display = "block";
                        exportBtn.innerText = "Download FBX";
                        exportBtn.title = "Download the original FBX file";
                    } else {
                        exportBtn.style.display = "block";
                        exportBtn.innerText = "Export GLB";
                        exportBtn.title = "Export selection as GLB (3D Mesh)";
                    }
                } else {
                    statusLabel.innerText = `Selected: ${count} objects (Ctrl+Click to add)`;
                    statusLabel.style.color = "#0f0";
                    exportBtn.style.display = "block";
                    if (isLegacyPlayer) {
                        exportBtn.innerText = `Download ${count} FBXs`;
                        exportBtn.title = "Download all selected FBX files";
                    } else {
                        exportBtn.innerText = `Export ${count} GLBs`;
                        exportBtn.title = "Export all selected GLB files";
                    }
                }
            };

            const removeSelectionAt = (idx) => {
                const selection = selectedModels[idx];
                if (selection.highlight) scene.remove(selection.highlight);

                if (selection.type === "model") {
                    selection.obj.model.traverse((child) => {
                        if (child.isMesh) {
                            const mats = Array.isArray(child.material) ? child.material : [child.material];
                            for (const m of mats) {
                                if (!m) continue;
                                if (m.userData && m.userData.origEmissive) m.emissive.copy(m.userData.origEmissive);
                                else m.emissive = new THREE.Color(0, 0, 0);
                                m.emissiveIntensity = 0;
                            }
                        }
                    });
                } else if (selection.type === "skeleton") {
                    for (const joint of selection.obj.joints) {
                        joint.material.emissiveIntensity = 0.3;
                    }
                }
                selectedModels.splice(idx, 1);
            };

            const focusOn = (obj) => {
                if (!obj || !orbitControls) return;
                const box = new THREE.Box3().setFromObject(obj);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const targetPos = new THREE.Vector3(center.x, center.y + (maxDim * 0.2), center.z + Math.max(maxDim * 2.5, 3));

                const startPos = camera.position.clone();
                const startTarget = orbitControls.target.clone();
                let t = 0;
                const anim = () => {
                    t += 0.05;
                    if (t <= 1.0) {
                        camera.position.lerpVectors(startPos, targetPos, t);
                        orbitControls.target.lerpVectors(startTarget, center, t);
                        requestAnimationFrame(anim);
                    }
                };
                anim();
            };

            const deselectAll = () => {
                while (selectedModels.length > 0) {
                    removeSelectionAt(0);
                }
                updateUI();
            };

            const exportSelected = async () => {
                if (selectedModels.length === 0) {
                    alert("Please select at least one object to export.");
                    return;
                }

                if (nodeData.name === "HYMotionFBXPlayer") {
                    // Direct Download Logic for Legacy FBX Player - Support Multi-Selection
                    for (const selection of selectedModels) {
                        if (selection.type !== "model") continue;

                        const { fileName, subfolder, fileType } = selection.obj;

                        let fetchUrl = `${window.location.origin}/view?type=${encodeURIComponent(fileType || 'output')}&filename=${encodeURIComponent(fileName)}`;
                        if (subfolder) fetchUrl += `&subfolder=${encodeURIComponent(subfolder)}`;

                        const link = document.createElement('a');
                        link.href = fetchUrl;
                        link.download = fileName;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);

                        // Add a small delay between downloads to prevent browser blocking
                        if (selectedModels.length > 1) {
                            await new Promise(resolve => setTimeout(resolve, 100));
                        }
                    }
                    return;
                }

                const { GLTFExporter } = window.__HY_MOTION_EXPORTER__;
                if (!GLTFExporter) {
                    alert("GLTFExporter not loaded yet.");
                    return;
                }

                const exporter = new GLTFExporter();

                for (let i = 0; i < selectedModels.length; i++) {
                    const selection = selectedModels[i];
                    let objToExport;
                    if (selection.type === "model") objToExport = selection.obj.model;
                    else if (selection.type === "skeleton") objToExport = selection.obj.group;

                    if (!objToExport) continue;

                    let filename = (selection.obj.name || (selection.type + "_" + i)) + ".glb";
                    console.log(`[HY-Motion] Exporting ${i + 1}/${selectedModels.length}:`, filename);

                    // Clone to reset highlights for export
                    const clone = objToExport.clone();
                    clone.traverse(child => {
                        if (child.isMesh && child.material) {
                            const mats = Array.isArray(child.material) ? child.material : [child.material];
                            mats.forEach(m => {
                                if (m.userData && m.userData.origEmissive) m.emissive.copy(m.userData.origEmissive);
                                else if (m.emissive) m.emissive.set(0, 0, 0);
                                m.emissiveIntensity = 0;
                            });
                        }
                    });

                    // Trigger export
                    await new Promise((resolve) => {
                        exporter.parse(
                            clone,
                            (result) => {
                                const blob = new Blob([result], { type: 'application/octet-stream' });
                                const link = document.createElement('a');
                                link.href = URL.createObjectURL(blob);
                                link.download = filename;
                                link.click();
                                setTimeout(resolve, 300);
                            },
                            (error) => {
                                console.error("[HY-Motion] Export failed:", error);
                                resolve();
                            },
                            { binary: true, animations: [] }
                        );
                    });
                }
                console.log("[HY-Motion] Multi-export sequence complete.");
            };

            // Helper to start animation loop
            const startAnimating = () => {
                if (!isAnimating) {
                    isAnimating = true;
                    animate();
                }
            };

            // Helper to stop animation loop
            const stopAnimating = () => {
                isAnimating = false;
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }
            };

            // Helper to request a single render
            const requestRender = () => {
                needsRender = true;
                if (!isAnimating) startAnimating();
            };

            const animate = () => {
                // Only continue loop if we should be animating
                if (!isAnimating) {
                    animationFrameId = null;
                    return;
                }

                animationFrameId = requestAnimationFrame(animate);
                if (!THREE) return; // Guard against early frames

                const delta = clock ? clock.getDelta() : 0.016;
                let shouldRender = needsRender;

                if (isPlaying && maxFrames > 0 && !isScrubbing) {
                    shouldRender = true; // Always render when playing

                    // Update current frame (float for sub-frame interpolation)
                    frameAccumulator += delta;
                    // Wrap accumulator to loop correctly
                    const duration = frameTime * maxFrames;
                    if (frameAccumulator >= duration) {
                        frameAccumulator %= duration;
                    }
                    if (frameAccumulator < 0) frameAccumulator = 0;

                    currentFrame = frameAccumulator / frameTime;

                    // Sync mixers
                    if (mixers.length > 0) {
                        const loopTime = maxFrames * frameTime;
                        for (const m of mixers) {
                            m.setTime(frameAccumulator % loopTime);
                        }
                    }

                    // Sync skeleton view with interpolation (if any)
                    updateSkeletons(currentFrame);

                    // Update progress UI (only every ~5 frames to reduce overhead)
                    if (!isScrubbing && Math.floor(currentFrame) % 3 === 0) {
                        const nextVal = (currentFrame / maxFrames) * 100;
                        if (Math.abs(progress.value - nextVal) > 0.5) {
                            progress.value = nextVal;
                        }
                    }
                } else {
                    // When not playing, only update if orbit controls have damping
                    if (orbitControls && orbitControls.enableDamping) {
                        shouldRender = true;
                    } else if (!needsRender) {
                        // Nothing to render and no damping, stop the loop
                        stopAnimating();
                        return;
                    }

                    // Only update hit proxies and highlights when NOT playing
                    // This saves significant CPU during animation playback
                    if (shouldRender) {
                        updateHitProxies();
                        updateSelectionHighlights();
                    }
                }

                if (orbitControls) orbitControls.update();

                // Only render if needed
                if (shouldRender && renderer && scene && camera) {
                    renderer.render(scene, camera);
                    needsRender = false; // Reset dirty flag
                }
            };


            // Utility to properly dispose of Three.js objects
            const disposeObject = (obj) => {
                if (!obj) return;
                obj.traverse((child) => {
                    if (child.geometry) child.geometry.dispose();
                    if (child.material) {
                        const materials = Array.isArray(child.material) ? child.material : [child.material];
                        materials.forEach((mat) => {
                            if (!mat) return;
                            // Dispose textures
                            for (const key in mat) {
                                if (mat[key] && mat[key].isTexture) mat[key].dispose();
                            }
                            mat.dispose();
                        });
                    }
                });
            };

            // Cache for bounding boxes to avoid recomputing every frame
            const bboxCache = new Map();

            const getCachedBBox = (obj) => {
                if (!bboxCache.has(obj.id)) {
                    bboxCache.set(obj.id, {
                        box: new THREE.Box3(),
                        lastUpdate: 0
                    });
                }
                const cache = bboxCache.get(obj.id);
                const now = Date.now();

                // Refresh cache if it's older than 100ms or hasn't been set
                if (now - cache.lastUpdate > 100) {
                    cache.box.copy(getRealtimeBox(obj));
                    cache.lastUpdate = now;
                }
                return cache.box;
            };

            // Separate function to update hit proxies (only when needed)
            const updateHitProxies = () => {
                const now = Date.now();
                for (const obj of loadedModels) {
                    if (!obj.hitProxy) {
                        const geo = new THREE.BoxGeometry(1, 1, 1);
                        const mat = new THREE.MeshBasicMaterial({
                            visible: false,
                            wireframe: true,
                            side: THREE.DoubleSide
                        });
                        const proxy = new THREE.Mesh(geo, mat);
                        proxy.userData.selectableParent = obj;
                        proxy.userData.type = "model";
                        scene.add(proxy);
                        obj.hitProxy = proxy;
                    }

                    // Only update at 10fps for performance
                    if (!obj._lastProxyUpdate || now - obj._lastProxyUpdate > 100) {
                        const box = getCachedBBox(obj.model);
                        if (!box.isEmpty()) {
                            const size = box.getSize(new THREE.Vector3());
                            const center = box.getCenter(new THREE.Vector3());
                            obj.hitProxy.scale.set(size.x * 1.3, size.y * 1.1, size.z * 1.3);
                            obj.hitProxy.position.copy(center);
                            obj._lastProxyUpdate = now;
                        }
                    }
                }
            };

            // Separate function to update selection highlights (only when needed)
            const updateSelectionHighlights = () => {
                const now = Date.now();
                for (const selection of selectedModels) {
                    if (selection.highlight) {
                        if (selection.type === "model") {
                            // Only update at 10fps for performance
                            if (!selection._lastHighlightUpdate || now - selection._lastHighlightUpdate > 100) {
                                const box = getCachedBBox(selection.obj.model);
                                if (!box.isEmpty()) {
                                    const size = box.getSize(new THREE.Vector3());
                                    const center = box.getCenter(new THREE.Vector3());
                                    selection.highlight.scale.set(size.x, size.y, size.z);
                                    selection.highlight.position.copy(center);
                                    selection._lastHighlightUpdate = now;
                                }
                            }
                        } else {
                            selection.highlight.update();
                        }
                    }
                }
            };

            const clearModels = () => {
                mixers.forEach(m => m.stopAllAction());
                mixers = [];
                loadedModels.forEach(m => {
                    scene.remove(m.model);
                    disposeObject(m.model); // Dispose resources
                    if (m.hitProxy) {
                        scene.remove(m.hitProxy);
                        disposeObject(m.hitProxy);
                    }
                });
                loadedModels = [];
                currentModel = null;
                bboxCache.clear(); // Clear cache when models are gone
                deselectAll();
                statusLabel.innerText = "Ready";
            };

            const clearSkeletons = () => {
                for (const s of skeletalSamples) {
                    scene.remove(s.group);
                    disposeObject(s.group); // Dispose resources
                }
                skeletalSamples = [];
                maxFrames = 0;
            };

            const clearScene = () => {
                console.log("[HY-Motion] Clearing entire scene...");
                clearModels();
                clearSkeletons();
                isPlaying = false;
                playBtn.innerText = "Play";
            };

            const centerCameraOnObject = (obj) => {
                if (!obj) return;
                setTimeout(() => {
                    const box = new THREE.Box3().setFromObject(obj);
                    if (box.isEmpty()) return;
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    orbitControls.target.copy(center);
                    camera.position.set(center.x, center.y + (maxDim * 0.2), center.z + Math.max(maxDim * 2.5, 4));
                    orbitControls.update();
                }, 150);
            };

            const loadMotions = (motions) => {
                if (!Array.isArray(motions)) return;
                console.log("[HY-Motion] Loading Skeletons:", motions.length);
                clearSkeletons();
                const masterGroup = new THREE.Group();
                let allMaxFrames = 0;
                for (const m of motions) {
                    if (!m.keypoints || !m.keypoints[0]) continue;
                    const group = new THREE.Group();
                    group.position.x = (m.posX || 0) - (motions.length - 1) * 0.75;
                    const sampleColor = m.color || 0x3366ff;
                    const jointMat = new THREE.MeshStandardMaterial({ color: sampleColor, emissive: sampleColor, emissiveIntensity: 0.3 });
                    const boneMat = new THREE.LineBasicMaterial({ color: sampleColor, transparent: true, opacity: 0.8 });
                    const joints = [];
                    const firstFrame = m.keypoints[0];
                    const numJoints = firstFrame.length;
                    for (let j = 0; j < numJoints; j++) {
                        const size = j === 0 ? 0.12 : 0.05; // Slightly larger for easier clicking
                        const sph = new THREE.Mesh(new THREE.SphereGeometry(size, 12, 12), jointMat);
                        group.add(sph);
                        joints.push(sph);
                    }
                    const bones = [];
                    for (const [j1, j2] of SMPL_H_SKELETON) {
                        if (j1 < numJoints && j2 < numJoints) {
                            const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
                            const line = new THREE.Line(geo, boneMat);
                            group.add(line);
                            bones.push({ line, j1, j2 });
                        }
                    }
                    masterGroup.add(group);
                    skeletalSamples.push({ group, joints, bones, keypoints: m.keypoints });
                    allMaxFrames = Math.max(allMaxFrames, m.keypoints.length);
                }
                scene.add(masterGroup);
                maxFrames = allMaxFrames;
                currentFrame = 0;
                frameAccumulator = 0; // Reset accumulator
                updateSkeletons(0);
                isPlaying = true;
                playBtn.innerText = "Pause";
                statusLabel.innerText = `${skeletalSamples.length} Skel${loadedModels.length ? ' + Model' : ''}`;
                // Only center if we don't have a model yet (model takes priority for centering)
                if (loadedModels.length === 0) centerCameraOnObject(masterGroup);
            };

            const updateSkeletons = (frameIdxFloat) => {
                for (const sample of skeletalSamples) {
                    const totalFrames = sample.keypoints.length;
                    if (totalFrames === 0) continue;

                    const f1 = Math.floor(frameIdxFloat) % totalFrames;
                    const f2 = (f1 + 1) % totalFrames;
                    const alpha = frameIdxFloat % 1;

                    const pos1 = sample.keypoints[f1];
                    const pos2 = sample.keypoints[f2];
                    if (!pos1 || !pos2) continue;

                    for (let i = 0; i < sample.joints.length; i++) {
                        const p1 = pos1[i];
                        const p2 = pos2[i];
                        if (p1 && p2) {
                            // Linear interpolation between frames
                            sample.joints[i].position.set(
                                p1[0] + (p2[0] - p1[0]) * alpha,
                                p1[1] + (p2[1] - p1[1]) * alpha,
                                p1[2] + (p2[2] - p1[2]) * alpha
                            );
                        }
                    }

                    for (const b of sample.bones) {
                        const j1p1 = pos1[b.j1], j1p2 = pos1[b.j2];
                        const j2p1 = pos2[b.j1], j2p2 = pos2[b.j2];

                        if (j1p1 && j1p2 && j2p1 && j2p2 && THREE) {
                            // Interpolate bone endpoints
                            const v1 = new THREE.Vector3(
                                j1p1[0] + (j2p1[0] - j1p1[0]) * alpha,
                                j1p1[1] + (j2p1[1] - j1p1[1]) * alpha,
                                j1p1[2] + (j2p1[2] - j1p1[2]) * alpha
                            );
                            const v2 = new THREE.Vector3(
                                j1p2[0] + (j2p2[0] - j1p2[0]) * alpha,
                                j1p2[1] + (j2p2[1] - j1p2[1]) * alpha,
                                j1p2[2] + (j2p2[2] - j1p2[2]) * alpha
                            );
                            b.line.geometry.setFromPoints([v1, v2]);
                        }
                    }
                }
            };

            const loadGenericModel = async (modelPath, format, customName = null, index = 0, total = 1) => {
                if (!modelPath || modelPath === "none") return;

                // Ensure Three is loaded if called from widget before initThree finished
                if (!THREE && window.__HY_MOTION_THREE__) THREE = window.__HY_MOTION_THREE__;
                if (!THREE) {
                    console.log("[HY-Motion] Three not ready, waiting for init...");
                    await initThree();
                }

                console.log("[HY-Motion] Loading 3D Model Path:", modelPath, `(${index + 1}/${total})`);

                // Optimization: Check if this model is already loaded
                if (total === 1 && loadedModels.length === 1 && loadedModels[0].modelPath === modelPath) {
                    console.log("[HY-Motion] Model already loaded, skipping redundant reload.");
                    refreshTransforms();
                    return;
                }

                if (index === 0) {
                    clearModels();
                    statusLabel.innerText = "Loading Model...";
                }

                try {
                    let loaderModule = null;
                    let LoaderClass = null;

                    if (format === 'fbx') {
                        if (!window.__HY_MOTION_FBX_LOADER__) window.__HY_MOTION_FBX_LOADER__ = await import(FBX_LOADER_URL);
                        if (!window.fflate) window.fflate = await import("https://esm.sh/fflate@0.8.0");
                        LoaderClass = window.__HY_MOTION_FBX_LOADER__.FBXLoader;
                    } else if (format === 'glb' || format === 'gltf') {
                        if (!window.__HY_MOTION_GLTF_LOADER__) window.__HY_MOTION_GLTF_LOADER__ = await import(GLTF_LOADER_URL);
                        LoaderClass = window.__HY_MOTION_GLTF_LOADER__.GLTFLoader;
                    } else if (format === 'obj') {
                        if (!window.__HY_MOTION_OBJ_LOADER__) window.__HY_MOTION_OBJ_LOADER__ = await import(OBJ_LOADER_URL);
                        LoaderClass = window.__HY_MOTION_OBJ_LOADER__.OBJLoader;
                    }

                    if (!LoaderClass) throw new Error("Unsupported format: " + format);
                    const loader = new LoaderClass();
                    if (format === 'fbx' && window.fflate && typeof loader.setFflate === 'function') {
                        loader.setFflate(window.fflate);
                    }

                    // Parse folder and filename
                    let type = "output";
                    let filename = modelPath;
                    let subfolder = "";

                    if (modelPath.startsWith("input/")) {
                        type = "input";
                        filename = modelPath.substring(6);
                    } else if (modelPath.startsWith("output/")) {
                        type = "output";
                        filename = modelPath.substring(7);
                    }

                    if (filename.includes("/") || filename.includes("\\")) {
                        const parts = filename.split(/[/\\]/);
                        filename = parts.pop();
                        subfolder = parts.join("/");
                    }

                    let fetchUrl = `${window.location.origin}/view?type=${type}&filename=${encodeURIComponent(filename)}`;
                    if (subfolder) fetchUrl += `&subfolder=${encodeURIComponent(subfolder)}`;

                    const loadingLabel = customName || filename;
                    statusLabel.innerText = `Loading ${loadingLabel}...`;
                    statusLabel.style.color = "#ffaa00";

                    console.log("[HY-Motion] Loading 3D Model:", fetchUrl);

                    // Progress callback to show loading progress
                    const onProgress = (xhr) => {
                        if (xhr.lengthComputable) {
                            const pct = Math.round((xhr.loaded / xhr.total) * 100);
                            statusLabel.innerText = `Loading ${loadingLabel}... ${pct}%`;
                        }
                    };

                    loader.load(fetchUrl, (result) => {
                        // Defer processing to next frame to avoid blocking UI
                        requestAnimationFrame(() => {
                            // Double check THREE is available in callback scope
                            const T = THREE || window.__HY_MOTION_THREE__;
                            if (!T) { console.error("[HY-Motion] THREE still null in callback!"); return; }


                            const fbx = (format === 'glb' || format === 'gltf') ? result.scene : result;
                            fbx.updateMatrixWorld(true);

                            const debugMat = new T.MeshStandardMaterial({ color: 0xffaa00, roughness: 0.5, metalness: 0.5, side: T.DoubleSide });
                            fbx.traverse((child) => {
                                if (child.isMesh) {
                                    if (!child.material || child.material.type === "MeshBasicMaterial") child.material = debugMat;
                                    child.castShadow = true;
                                    child.receiveShadow = true;
                                    child.frustumCulled = false;
                                }
                            });

                            scene.add(fbx);
                            currentModel = fbx;
                            loadedModels.push({
                                model: fbx,
                                modelPath: modelPath,
                                name: customName || filename,
                                fileName: filename,
                                subfolder: subfolder,
                                fileType: type,
                                basePosition: { x: 0, y: 0, z: 0 },
                                baseRotation: { x: 0, y: 0, z: 0 },
                                baseScale: { x: 1, y: 1, z: 1 }
                            });

                            // Positioning side-by-side
                            if (total > 1) {
                                fbx.position.x = index * 1.5 - (total - 1) * 0.75;
                            }

                            const box = new THREE.Box3().setFromObject(fbx);
                            const maxDim = Math.max(...box.getSize(new THREE.Vector3()).toArray());
                            if (maxDim > 0) fbx.scale.setScalar(1.7 / maxDim);

                            // Update base transforms to match finalized load state
                            const obj = loadedModels.find(m => m.model === fbx);
                            if (obj) {
                                obj.basePosition.x = fbx.position.x;
                                obj.basePosition.y = fbx.position.y;
                                obj.basePosition.z = fbx.position.z;
                                obj.baseRotation.x = fbx.rotation.x * 180 / Math.PI;
                                obj.baseRotation.y = fbx.rotation.y * 180 / Math.PI;
                                obj.baseRotation.z = fbx.rotation.z * 180 / Math.PI;
                                obj.baseScale.x = fbx.scale.x;
                                obj.baseScale.y = fbx.scale.y;
                                obj.baseScale.z = fbx.scale.z;
                            }

                            if (fbx.animations && fbx.animations.length > 0 || (result.animations && result.animations.length > 0)) {
                                const m = new THREE.AnimationMixer(fbx);
                                const anims = fbx.animations || result.animations;
                                const action = m.clipAction(anims[0]);
                                action.play();
                                mixers.push(m);

                                // If we don't have skeletal maxFrames, use the animation duration
                                if (maxFrames === 0) {
                                    maxFrames = Math.floor(anims[0].duration * targetFPS);
                                    if (maxFrames === 0) maxFrames = 1; // Fallback
                                }

                                // Only auto-play if it's NOT the standalone 3D Model Loader
                                if (nodeData.name !== "HYMotion3DModelLoader") {
                                    isPlaying = true;
                                    playBtn.innerText = "Pause";
                                    startAnimating(); // Start animation loop
                                }
                                frameAccumulator = 0; // Reset accumulator
                            }

                            statusLabel.innerText = total > 1 ? `Loaded ${loadedModels.length}/${total}` : "Model Loaded";

                            // Initialize hit proxies for the new model
                            updateHitProxies();
                            requestRender(); // Trigger render

                            if (total > 1) {
                                // Center on all models
                                const group = new THREE.Group();
                                loadedModels.forEach(m => group.add(m.model.clone())); // Dummy group for centering auto-box
                                centerCameraOnObject(group);
                            } else {
                                centerCameraOnObject(fbx);
                            }

                            // Show Transform and Apply buttons for loader
                            if (nodeData.name === "HYMotion3DModelLoader") {
                                applyBtn.style.display = "block";
                                transformBtn.style.display = "block";
                            }

                            // Apply current transform widget values immediately
                            refreshTransforms();
                        }); // Close requestAnimationFrame
                    }, onProgress, (err) => {
                        console.error("[HY-Motion] Load fail:", err);
                        statusLabel.innerText = "Load Error";
                        statusLabel.style.color = "red";
                    });


                } catch (e) {
                    console.error("3D Load Error:", e);
                    statusLabel.innerText = "Error: " + e.message;
                    statusLabel.style.color = "red";
                }
            };

            const ensureString = (val) => {
                if (typeof val === 'string') return val;
                if (Array.isArray(val)) {
                    // Check if it's a character array (e.g. ['f', 'b', 'x'])
                    if (val.every(i => typeof i === 'string' && i.length <= 1)) return val.join('');
                    // Otherwise take the last element (ComfyUI pattern)
                    return ensureString(val[val.length - 1]);
                }
                return String(val);
            };

            const handleData = async (data) => {
                if (!data) return;

                // Lazy init - only load Three.js when we actually have data to display
                await ensureInitialized();

                if (data.motions) {
                    // Use a hash or simply compare array lengths/first few elements if possible, 
                    // but for now, we'll keep the string compare but ensure we don't hold multiple copies.
                    const motionsStr = typeof data.motions === 'string' ? data.motions : JSON.stringify(data.motions);
                    if (motionsStr !== lastMotionsData) {
                        lastMotionsData = motionsStr;
                        try { loadMotions(JSON.parse(motionsStr)); }
                        catch (e) { console.error("[HY-Motion] JSON parse error:", e); }
                    }
                }

                // Handle both legacy 'fbx_url' and new 'fbx_paths'
                const rawUrl = data.fbx_url || data.fbx_paths;
                if (rawUrl) {
                    const urlStr = ensureString(rawUrl);
                    if (urlStr !== lastFbxUrl) {
                        lastFbxUrl = urlStr;
                        // Split by newline and load each model (usually just one, but supports multi-batch)
                        const urls = urlStr.split('\n').map(u => u.trim()).filter(u => u.length > 0);
                        urls.forEach((url, i) => {
                            console.log("[HY-Motion] Found FBX to load:", url);
                            // Extract filename for naming
                            const name = url.split(/[/\\]/).pop();
                            loadGenericModel(url, 'fbx', name, i, urls.length);
                        });
                    }
                }

                if (data.model_url) {
                    const url = ensureString(data.model_url);
                    const format = ensureString(data.format || 'fbx');
                    await loadGenericModel(url, format);
                }

                // Also refresh visuals if transform changed via Run
                refreshTransforms();
            };

            // Lazy initialization - only init when data is actually loaded
            // This prevents the 4-second freeze on ComfyUI reload
            const ensureInitialized = async () => {
                if (!isInitialized && !isInitializing) {
                    await initThree();
                }
            };

            // Support live preview on widget change for the loader nodes
            if (nodeData.name === "HYMotion3DModelLoader" || nodeData.name === "HYMotionFBXPlayer") {
                const widgetName = nodeData.name === "HYMotion3DModelLoader" ? "model_path" : "fbx_name";
                setTimeout(() => {
                    const modelWidget = this.widgets?.find(w => w.name === widgetName);
                    if (modelWidget) {
                        const oldCallback = modelWidget.callback;
                        modelWidget.callback = async function (value) {
                            if (oldCallback) oldCallback.apply(this, arguments);
                            console.log("[HY-Motion] Widget changed:", value);

                            // Lazy init on widget change
                            await ensureInitialized();

                            let finalUrl = value;
                            if (nodeData.name === "HYMotionFBXPlayer" && value !== "none") {
                                // Add "output/" prefix for legacy player if not present
                                finalUrl = value.startsWith("output/") ? value : `output/${value}`;
                            }

                            const ext = finalUrl.split('.').pop().toLowerCase();
                            loadGenericModel(finalUrl, ext);
                        };
                        // Trigger initial load
                        if (modelWidget.value && modelWidget.value !== "none") {
                            let finalUrl = modelWidget.value;
                            if (nodeData.name === "HYMotionFBXPlayer") {
                                finalUrl = finalUrl.startsWith("output/") ? finalUrl : `output/${finalUrl}`;
                            }
                            const ext = finalUrl.split('.').pop().toLowerCase();
                            loadGenericModel(finalUrl, ext);
                        }
                    }
                }, 500);
            }

            node.onExecuted = function (output) {
                // handleData is now async and handles lazy initialization
                handleData(output);
            };

            // Add real-time callbacks and hide transform widgets from Node UI
            const hideTransformWidgets = () => {
                const transformWidgets = [
                    "translate_x", "translate_y", "translate_z",
                    "rotate_x", "rotate_y", "rotate_z",
                    "scale_x", "scale_y", "scale_z"
                ];
                transformWidgets.forEach(wName => {
                    const w = node.widgets?.find(widget => widget.name === wName);
                    if (w) {
                        // Aggressive hiding strategy for ComfyUI
                        w.type = "hidden";
                        w.hidden = true;
                        if (!w.computeSize) {
                            w.computeSize = () => [0, -4]; // Standard ComfyUI trick for hidden widgets
                        }

                        // Keep the sync callback logic
                        if (!w._sync_callback_added) {
                            const oldCb = w.callback;
                            w.callback = function () {
                                if (oldCb) oldCb.apply(this, arguments);
                                refreshTransforms();
                            };
                            w._sync_callback_added = true;
                        }
                    }
                });
                if (app.graph) app.graph.setDirtyCanvas(true);
            };

            // Run hiding immediately and also with delay to catch post-init widgets
            hideTransformWidgets();
            setTimeout(hideTransformWidgets, 500);
            setTimeout(hideTransformWidgets, 1500);

            playBtn.onclick = (e) => {
                e.stopPropagation();
                isPlaying = !isPlaying;
                playBtn.innerText = isPlaying ? "Pause" : "Play";
                if (isPlaying) startAnimating(); // Start loop when playing
            };
            cycleBtn.onclick = (e) => { e.stopPropagation(); cycleSelection(); };
            exportBtn.onclick = (e) => { e.stopPropagation(); exportSelected(); };

            applyBtn.onclick = async (e) => {
                e.stopPropagation();
                if (!currentModel) return;

                const obj = loadedModels.find(m => m.model === currentModel);
                if (!obj) return;

                // Gather current transform values from widgets 
                const tx = node.widgets?.find(w => w.name === "translate_x")?.value || 0;
                const ty = node.widgets?.find(w => w.name === "translate_y")?.value || 0;
                const tz = node.widgets?.find(w => w.name === "translate_z")?.value || 0;
                const rx = node.widgets?.find(w => w.name === "rotate_x")?.value || 0;
                const ry = node.widgets?.find(w => w.name === "rotate_y")?.value || 0;
                const rz = node.widgets?.find(w => w.name === "rotate_z")?.value || 0;
                const sx = node.widgets?.find(w => w.name === "scale_x")?.value || 1;
                const sy = node.widgets?.find(w => w.name === "scale_y")?.value || 1;
                const sz = node.widgets?.find(w => w.name === "scale_z")?.value || 1;

                statusLabel.innerText = "Baking Asset...";
                statusLabel.style.color = "#ffcc00";

                try {
                    const response = await api.fetchApi("/hymotion/bake_fbx", {
                        method: "POST",
                        body: JSON.stringify({
                            input_path: obj.modelPath,
                            translation: [tx, ty, tz],
                            rotation: [rx, ry, rz],
                            scale: [sx, sy, sz]
                        })
                    });

                    if (!response.ok) {
                        const errData = await response.json();
                        throw new Error(errData.message || "Bake failed");
                    }

                    const resData = await response.json();
                    console.log("[HY-Motion] Asset Baked Successfully:", resData.path);
                    statusLabel.innerText = "Bake Success!";
                    statusLabel.style.color = "#00ff00";

                    // Update base transforms in memory (since the file is now physically changed)
                    obj.basePosition.x = currentModel.position.x;
                    obj.basePosition.y = currentModel.position.y;
                    obj.basePosition.z = currentModel.position.z;
                    obj.baseRotation.x = currentModel.rotation.x * 180 / Math.PI;
                    obj.baseRotation.y = currentModel.rotation.y * 180 / Math.PI;
                    obj.baseRotation.z = currentModel.rotation.z * 180 / Math.PI;
                    obj.baseScale.x = currentModel.scale.x;
                    obj.baseScale.y = currentModel.scale.y;
                    obj.baseScale.z = currentModel.scale.z;

                    // Reset ComfyUI widgets to zero (as the transform is now 'baked' into the base)
                    const transformWidgets = [
                        "translate_x", "translate_y", "translate_z",
                        "rotate_x", "rotate_y", "rotate_z"
                    ];
                    transformWidgets.forEach(wName => {
                        const w = node.widgets?.find(widget => widget.name === wName);
                        if (w) w.value = 0;
                    });
                    ["scale_x", "scale_y", "scale_z"].forEach(wName => {
                        const w = node.widgets?.find(widget => widget.name === wName);
                        if (w) w.value = 1.0;
                    });

                    if (app.graph) app.graph.setDirtyCanvas(true);

                    // Sync panel back to 0
                    const pInputs = transformOverlay.querySelectorAll('input');
                    pInputs.forEach(input => {
                        if (input.dataset.key.startsWith('scale')) input.value = "1.00";
                        else input.value = "0.00";
                    });

                    // Small delay to show "Success" then revert status
                    setTimeout(() => {
                        statusLabel.innerText = "Model Ready";
                        statusLabel.style.color = "white";
                    }, 2000);

                } catch (e) {
                    console.error("[HY-Motion] Bake Error:", e);
                    statusLabel.innerText = "Bake Error: " + e.message;
                    statusLabel.style.color = "red";
                }
            };
            progress.oninput = () => {
                isPlaying = false; playBtn.innerText = "Play";
                currentFrame = (progress.value / 100) * maxFrames;
                frameAccumulator = currentFrame * frameTime;

                // Sync mixers for FBX animations
                if (mixers.length > 0) {
                    for (const m of mixers) {
                        m.setTime(frameAccumulator % (maxFrames * frameTime));
                    }
                }

                // Sync skeletons for motion data
                updateSkeletons(currentFrame);
                requestRender(); // Trigger single render
            };

            // Proper cleanup when node is removed
            this.onRemoved = function () {
                console.log("[HY-Motion] Removing 3D Viewer Node and cleaning up resources:", node.id);

                // Stop animation
                stopAnimating();

                // Remove observers
                if (this._visibilityObserver) {
                    this._visibilityObserver.disconnect();
                    this._visibilityObserver = null;
                }
                if (this._resizeObserver) {
                    this._resizeObserver.disconnect();
                    this._resizeObserver = null;
                }

                // Remove global event listeners
                if (this._handleKeyPress) {
                    document.removeEventListener('keydown', this._handleKeyPress);
                    this._handleKeyPress = null;
                }

                // Dispose Three.js resources
                if (scene) {
                    scene.traverse((object) => {
                        disposeObject(object);
                    });
                }
                if (renderer) {
                    renderer.dispose();
                    if (renderer.domElement && renderer.domElement.parentNode) {
                        renderer.domElement.parentNode.removeChild(renderer.domElement);
                    }
                    renderer = null;
                }

                bboxCache.clear();
                scene = null;
                camera = null;
                orbitControls = null;
            };

            return r;
        };

        // Handle node resize to sync container and renderer
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            // Call original onResize if it exists
            if (onResize) {
                onResize.apply(this, arguments);
            }

            // Find the container widget
            const viewerWidget = this.widgets?.find(w => w.name === "3d_viewer");
            if (!viewerWidget || !viewerWidget.element) return;

            const container = viewerWidget.element;
            const canvasContainer = container.querySelector('div[style*="flex:1"]');

            if (!container || !canvasContainer) return;

            // Save the height preference for next load
            localStorage.setItem('hymotion_viewer_height', size[1] - 90);

            // Update renderer and camera if they exist
            // Use a small delay to ensure DOM has updated
            setTimeout(() => {
                const THREE = window.__HY_MOTION_THREE__;
                if (THREE && canvasContainer.clientWidth > 0 && canvasContainer.clientHeight > 0) {
                    // Find the canvas element
                    const canvas = canvasContainer.querySelector('canvas');
                    if (canvas) {
                        // Get the renderer from window storage (we'll need to store it)
                        const renderer = canvas.__renderer;
                        const camera = canvas.__camera;

                        if (renderer && camera) {
                            camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
                            camera.updateProjectionMatrix();
                            renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight, false);
                        }
                    }
                }
            }, 0);
        };
    }
});
