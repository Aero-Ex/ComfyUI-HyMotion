// Three.js Local Paths
const THREE_URL = "./lib/three/three.module.js";
const FBX_LOADER_URL = "./lib/three/examples/jsm/loaders/FBXLoader.js";
const ORBIT_CONTROLS_URL = "./lib/three/examples/jsm/controls/OrbitControls.js";
const TRANSFORM_CONTROLS_URL = "./lib/three/examples/jsm/controls/TransformControls.js";
const GLTF_EXPORTER_URL = "./lib/three/examples/jsm/exporters/GLTFExporter.js";
const GLTF_LOADER_URL = "./lib/three/examples/jsm/loaders/GLTFLoader.js";
const OBJ_LOADER_URL = "./lib/three/examples/jsm/loaders/OBJLoader.js";

// Global initialization state to prevent redundant library loads
let isGlobalInitializing = false;
let globalInitPromise = null;

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";


console.log("[HY-Motion] app imported successfully:", !!app);

// Track active viewer instances to manage WebGL context lifecycle
const activeViewerNodes = new Set();

// Global asset refresh function for all HY-Motion nodes
const refreshAllHyMotionAssets = async () => {
    try {
        console.log("[HY-Motion] Refreshing asset lists for all relevant nodes...");
        const response = await api.fetchApi("/hymotion/get_assets");
        if (!response.ok) return;
        const assets = await response.json();

        if (!app.graph || !app.graph._nodes) return;

        app.graph._nodes.forEach(node => {
            const nodeName = node.type;
            if (nodeName === "HYMotionFBXPlayer" || nodeName === "HYMotion3DModelLoader") {
                const widgetName = nodeName === "HYMotion3DModelLoader" ? "model_path" : "fbx_name";
                const widget = node.widgets?.find(w => w.name === widgetName);

                if (widget && assets) {
                    const newList = nodeName === "HYMotion3DModelLoader" ?
                        assets.fbx_files :
                        assets.fbx_files.filter(f => f.startsWith("output/")).map(f => f.substring(7));

                    widget.options.values = newList;
                }
            }
        });
        if (app.graph) app.graph.setDirtyCanvas(true);
    } catch (e) {
        console.error("[HY-Motion] Failed to refresh global assets:", e);
    }
};


app.registerExtension({
    name: "HYMotion.3DViewer",
    async init() {
        console.log("[HY-Motion] 3D Viewer Extension Loaded (v3.5 - Alt+Z Undo)");
        // Viewer undo/redo uses Alt+Z / Alt+Y to avoid conflict with ComfyUI's Ctrl+Z
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "HYMotion3DViewer" &&
            nodeData.name !== "HYMotionFBXPlayer" &&
            nodeData.name !== "HYMotion3DModelLoader" &&
            nodeData.name !== "HYMotionRigManipulator" &&
            nodeData.name !== "HYMotionModularExportFBX" &&
            nodeData.name !== "HYMotionRetargetFBX") return;

        const isViewerNode = (nodeData.name !== "HYMotionModularExportFBX" && nodeData.name !== "HYMotionRetargetFBX");

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            console.log("[HY-Motion] Creating Node:", nodeData.name, node.id);

            if (!isViewerNode) {
                node.onExecuted = function (output) {
                    if (output && output.timestamp) refreshAllHyMotionAssets();
                };
                return r;
            }

            // Track this node instance as a viewer node
            activeViewerNodes.add(node.id);

            // Hide the internal JSON widgets
            const internalWidgets = ["start_pose_json", "end_pose_json"];
            setTimeout(() => {
                node.widgets?.forEach(w => {
                    if (internalWidgets.includes(w.name)) {
                        w.type = "hidden";
                    }
                });
                // Initial asset refresh on creation
                if (nodeData.name === "HYMotionFBXPlayer" || nodeData.name === "HYMotion3DModelLoader") {
                    refreshAllHyMotionAssets();
                }
            }, 1);

            // Main container with dynamic height
            const defaultHeight = 400;
            const storedHeight = localStorage.getItem('hymotion_viewer_height') || defaultHeight;

            const container = document.createElement("div");
            container.style.cssText = `width:100%; height:100%; background:#111; position:relative; display:flex; flex-direction:column; border:1px solid #333; border-radius:4px; overflow:hidden;`;

            const canvasContainer = document.createElement("div");
            canvasContainer.style.cssText = "flex:1; width:100%; height:100%; min-height:0; overflow:hidden; position:relative;";
            container.appendChild(canvasContainer);

            // Store container on node for ComfyUI/Extension compatibility
            this.container = container;

            // Drop Overlay for file uploads
            const dropOverlay = document.createElement("div");
            dropOverlay.style.cssText = "position:absolute; top:0; left:0; width:100%; height:100%; background:rgba(0,123,255,0.3); border:3px dashed #007bff; display:none; align-items:center; justify-content:center; z-index:5000; pointer-events:none; box-sizing:border-box;";
            dropOverlay.innerHTML = '<div style="background:rgba(0,0,0,0.7); padding:15px 25px; border-radius:10px; color:#fff; font-weight:bold; font-size:18px; pointer-events:none;">Drop 3D Model to Upload</div>';
            container.appendChild(dropOverlay);

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

            const focusBtn = document.createElement("button");
            focusBtn.innerText = "🎯";
            focusBtn.title = "Center camera on selection";
            focusBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:14px; background:#444; color:#fff; border:1px solid #666; border-radius:3px; font-weight:bold; flex-shrink:0;";

            const inPlaceBtn = document.createElement("button");
            inPlaceBtn.innerText = "🏃‍♂️";
            inPlaceBtn.title = "Toggle In-Place Animation (global when no selection, per-selection when selected)";
            inPlaceBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:14px; background:#444; color:#fff; border:1px solid #666; border-radius:3px; font-weight:bold; flex-shrink:0;";

            // selectionInPlaceBtn is now merged into inPlaceBtn - kept for backwards compatibility
            const selectionInPlaceBtn = document.createElement("button");
            selectionInPlaceBtn.style.display = "none"; // Hidden - functionality merged into inPlaceBtn

            // Rig Manipulator Controls
            const rigControls = document.createElement("div");
            rigControls.style.cssText = "display:none; gap:4px; border-left:1px solid #444; padding-left:6px; margin-left:6px; flex-shrink:0;";

            const poseModeBtn = document.createElement("button");
            poseModeBtn.innerText = "🦴 Pose";
            poseModeBtn.title = "Toggle Bone Posing Mode - Select bones by clicking on the model";
            poseModeBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#333; color:#aaa; border:1px solid #555; border-radius:3px; font-weight:bold;";

            const setStartBtn = document.createElement("button");
            setStartBtn.innerText = "Set Start";
            setStartBtn.title = "Capture current pose as Start Frame";
            // HIDDEN: Pose capture in development
            setStartBtn.style.cssText = "display:none; cursor:pointer; padding:6px 10px; font-size:13px; background:#442222; color:#fff; border:1px solid #663333; border-radius:3px; font-weight:bold;";

            const setEndBtn = document.createElement("button");
            setEndBtn.innerText = "Set End";
            setEndBtn.title = "Capture current pose as End Frame";
            // HIDDEN: Pose capture in development
            setEndBtn.style.cssText = "display:none; cursor:pointer; padding:6px 10px; font-size:13px; background:#224422; color:#fff; border:1px solid #336633; border-radius:3px; font-weight:bold;";

            const markStartBtn = document.createElement("button");
            markStartBtn.innerText = "Mark Start";
            markStartBtn.title = "Set current frame as start_frame_idx";
            markStartBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#444422; color:#fff; border:1px solid #666633; border-radius:3px; font-weight:bold; display:none;";

            const markEndBtn = document.createElement("button");
            markEndBtn.innerText = "Mark End";
            markEndBtn.title = "Set current frame as end_frame_idx";
            markEndBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#224444; color:#fff; border:1px solid #336666; border-radius:3px; font-weight:bold; display:none;";

            const xrayBtn = document.createElement("button");
            xrayBtn.innerText = "X-Ray";
            xrayBtn.title = "Toggle X-Ray Skeleton - See bones through mesh";
            xrayBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#333; color:#aaa; border:1px solid #555; border-radius:3px; font-weight:bold;";

            const showSkeletonBtn = document.createElement("button");
            showSkeletonBtn.innerText = "🦴 Show";
            showSkeletonBtn.title = "Toggle Skeleton Visibility";
            showSkeletonBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#0066cc; color:#fff; border:1px solid #555; border-radius:3px; font-weight:bold;";

            const resetPoseBtn = document.createElement("button");
            resetPoseBtn.innerText = "Reset Pose";
            resetPoseBtn.title = "Reset bones to rest pose";
            resetPoseBtn.style.cssText = "cursor:pointer; padding:6px 10px; font-size:13px; background:#444; color:#fff; border:1px solid #666; border-radius:3px; font-weight:bold;";

            rigControls.appendChild(poseModeBtn);
            rigControls.appendChild(xrayBtn);
            rigControls.appendChild(showSkeletonBtn);
            rigControls.appendChild(resetPoseBtn);
            rigControls.appendChild(setStartBtn);
            rigControls.appendChild(setEndBtn);
            rigControls.appendChild(markStartBtn);
            rigControls.appendChild(markEndBtn);

            let xraySkeleton = false;
            xrayBtn.onclick = () => {
                xraySkeleton = !xraySkeleton;
                xrayBtn.style.background = xraySkeleton ? "#0066cc" : "#333";
                xrayBtn.style.color = xraySkeleton ? "#fff" : "#aaa";
                updateBoneGizmos();
                requestRender();
            };

            let showSkeleton = true;
            showSkeletonBtn.onclick = () => {
                showSkeleton = !showSkeleton;
                showSkeletonBtn.style.background = showSkeleton ? "#0066cc" : "#333";
                showSkeletonBtn.style.color = showSkeleton ? "#fff" : "#aaa";
                updateBoneGizmos();
                requestRender();
            };

            resetPoseBtn.onclick = () => {
                const modelName = currentModel ? (currentModel.name || "unnamed") : "none";
                console.log(`[HY-Motion] Reset Pose: ${modelName}`);
                if (nodeData.name === "HYMotion3DModelLoader") saveStateForUndo();
                resetToRestPose();
                requestRender();
            };

            poseModeBtn.onclick = () => {
                isPoseMode = !isPoseMode;
                console.log(`[HY-Motion] Pose Mode: ${isPoseMode ? "ENABLED" : "DISABLED"} (loop will ${isPoseMode ? "stay active" : "sleep when idle"})`);
                poseModeBtn.style.background = isPoseMode ? "#0066cc" : "#333";
                poseModeBtn.style.color = isPoseMode ? "#fff" : "#aaa";

                if (isPoseMode && currentModel) {
                    createBoneGizmos(currentModel);
                } else {
                    // Clear gizmos when exiting pose mode
                    boneGizmos.forEach(g => scene.remove(g));
                    boneGizmos = [];
                    boneLines.forEach(l => scene.remove(l));
                    boneLines = [];
                    if (pivotIndicator) {
                        pivotIndicator.visible = false;
                    }
                    if (node.transformControl) {
                        node.transformControl.detach();
                    }
                    selectedBone = null;
                }
                updateBoneGizmos();
                requestRender();
            };

            setStartBtn.onclick = () => {
                const w = node.widgets.find(w => w.name === "start_pose_json");
                console.log("[HY-Motion] Capture Start Pose. Widget found:", !!w);
                const json = serializePose();
                if (w) {
                    w.value = json;
                    if (node.onWidgetChanged) node.onWidgetChanged(w.name, json, json, w);
                    statusLabel.innerText = "Start Pose Captured";
                    statusLabel.style.color = "#0f0";
                } else {
                    console.error("[HY-Motion] ERROR: start_pose_json widget not found!");
                    statusLabel.innerText = "Error: Widget missing";
                    statusLabel.style.color = "#f00";
                }
            };

            setEndBtn.onclick = () => {
                const w = node.widgets.find(w => w.name === "end_pose_json");
                console.log("[HY-Motion] Capture End Pose. Widget found:", !!w);
                const json = serializePose();
                if (w) {
                    w.value = json;
                    if (node.onWidgetChanged) node.onWidgetChanged(w.name, json, json, w);
                    statusLabel.innerText = "End Pose Captured";
                    statusLabel.style.color = "#0f0";
                } else {
                    console.error("[HY-Motion] ERROR: end_pose_json widget not found!");
                    statusLabel.innerText = "Error: Widget missing";
                    statusLabel.style.color = "#f00";
                }
            };

            markStartBtn.onclick = () => {
                const w = node.widgets.find(w => w.name === "start_frame");
                if (w) {
                    w.value = Math.floor(currentFrame);
                    if (node.onWidgetChanged) node.onWidgetChanged(w.name, w.value, w.value, w);
                }
                statusLabel.innerText = `Start Frame: ${Math.floor(currentFrame)}`;
                statusLabel.style.color = "#ff0";
            };

            markEndBtn.onclick = () => {
                const w = node.widgets.find(w => w.name === "end_frame");
                if (w) {
                    w.value = Math.floor(currentFrame);
                    if (node.onWidgetChanged) node.onWidgetChanged(w.name, w.value, w.value, w);
                }
                statusLabel.innerText = `End Frame: ${Math.floor(currentFrame)}`;
                statusLabel.style.color = "#0ff";
            };

            controls.appendChild(gizmoGroup);
            controls.appendChild(playBtn);
            controls.appendChild(inPlaceBtn);
            // selectionInPlaceBtn removed from UI - functionality merged into inPlaceBtn
            controls.appendChild(focusBtn);
            controls.appendChild(cycleBtn);
            controls.appendChild(exportBtn);
            controls.appendChild(rigControls);
            controls.appendChild(transformBtn);
            controls.appendChild(applyBtn);
            controls.appendChild(progress);
            controls.appendChild(statusLabel);
            container.appendChild(controls);

            if (nodeData.name === "HYMotion3DModelLoader") {
                rigControls.style.display = "flex";
            }
            if (nodeData.name === "HYMotionLoadNPZ") {
                markStartBtn.style.display = "block";
                markEndBtn.style.display = "block";
            }

            // Drag and Drop Handlers
            if (nodeData.name === "HYMotion3DModelLoader") {
                container.addEventListener("dragover", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropOverlay.style.display = "flex";
                });

                container.addEventListener("dragleave", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropOverlay.style.display = "none";
                });

                container.addEventListener("drop", async (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropOverlay.style.display = "none";

                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        const file = files[0];
                        const ext = file.name.split('.').pop().toLowerCase();
                        const allowed = ["fbx", "glb", "gltf", "obj"];

                        if (!allowed.includes(ext)) {
                            statusLabel.innerText = "Error: Invalid file type";
                            return;
                        }

                        statusLabel.innerText = "Uploading...";
                        try {
                            const formData = new FormData();
                            formData.append("image", file);
                            formData.append("overwrite", "true");
                            formData.append("type", "input");

                            const response = await api.fetchApi("/upload/image", {
                                method: "POST",
                                body: formData
                            });

                            if (response.status === 200 || response.status === 201) {
                                const data = await response.json();
                                const serverPath = `input/${data.name || data.filename || file.name}`;

                                // Update widget
                                const widget = node.widgets.find(w => w.name === "model_path");
                                if (widget) {
                                    // Add to options if not present
                                    if (!widget.options.values.includes(serverPath)) {
                                        widget.options.values.push(serverPath);
                                        widget.options.values.sort();
                                    }
                                    widget.value = serverPath;
                                }

                                statusLabel.innerText = "Uploaded: " + file.name;

                                // Trigger immediate load
                                if (isInitialized) {
                                    handleData({ model_url: serverPath, format: ext });
                                }
                            } else {
                                statusLabel.innerText = "Upload failed";
                            }
                        } catch (err) {
                            console.error("[HY-Motion] Upload Error:", err);
                            statusLabel.innerText = "Upload Error";
                        }
                    }
                });
            }

            // Transform Overlay Panel
            const transformPanel = document.createElement("div");
            transformPanel.style.cssText = "position:absolute; top:10px; right:10px; width:220px; background:rgba(20,20,20,0.85); backdrop-filter:blur(5px); border:1px solid #444; border-radius:8px; display:none; flex-direction:column; padding:12px; gap:10px; z-index:2000; color:#eee; font-family:sans-serif; user-select:none;";
            canvasContainer.appendChild(transformPanel);

            // Performance Tracing System
            const tracer = {
                lastLog: Date.now(),
                logInterval: 2000,
                active: false,
                frameCounts: 0,
                activeFrames: 0,
                diagElement: null,
                mark(name) { if (this.active) performance.mark(`${name}-${node.id}`); },
                measure(name, start, end) {
                    if (this.active) {
                        try { performance.measure(`${name}-${node.id}`, `${start}-${node.id}`, `${end}-${node.id}`); } catch (e) { }
                    }
                },
                report() {
                    this.frameCounts++;
                    if (!this.active) return;
                    const now = Date.now();
                    if (now - this.lastLog < this.logInterval) return;

                    const timeWindow = (now - this.lastLog) / 1000;
                    const fps = (this.frameCounts / timeWindow).toFixed(1);

                    const entries = performance.getEntriesByType("measure").filter(e => e.name.endsWith(`-${node.id}`));
                    const summary = {};
                    entries.forEach(e => {
                        const base = e.name.split('-')[0];
                        if (!summary[base]) summary[base] = { total: 0, count: 0, max: 0 };
                        summary[base].total += e.duration;
                        summary[base].count++;
                        summary[base].max = Math.max(summary[base].max, e.duration);
                    });

                    let out = `FPS: ${fps} | `;
                    for (const k in summary) {
                        const avg = (summary[k].total / summary[k].count).toFixed(2);
                        const max = summary[k].max.toFixed(2);
                        const isSlow = summary[k].max > 30;
                        const color = isSlow ? "#f00" : "#0f0";
                        out += `<span style="color:${color}">${k}: ${avg}(${max}pk)</span> | `;
                    }
                    if (this.diagElement) {
                        this.diagElement.innerHTML = out;
                        this.diagElement.style.display = "block";
                    }
                    console.log(`[HY-Node ${node.id}] ${out}`);

                    this.lastLog = now;
                    this.frameCounts = 0;
                    entries.forEach(e => performance.clearMeasures(e.name));
                    performance.clearMarks();
                }
            };
            window.__HY_TRACER__ = tracer;
            tracer.active = true;

            const diagPanel = document.createElement("div");
            diagPanel.style.cssText = "position:absolute; bottom:50px; left:10px; padding:8px; background:rgba(0,0,0,0.7); color:#0f0; font-family:monospace; font-size:10px; pointer-events:none; border-radius:4px; z-index:2001; display:none;";
            canvasContainer.appendChild(diagPanel);
            tracer.diagElement = diagPanel;

            // Diagnostics Toggle Button
            const diagBtn = document.createElement("button");
            diagBtn.innerText = "D";
            diagBtn.title = "Toggle Performance Diagnostics";
            diagBtn.style.cssText = "position:absolute; bottom:5px; left:5px; width:20px; height:20px; background:rgba(0,0,0,0.5); color:#0f0; border:none; border-radius:3px; cursor:pointer; font-size:10px; z-index:2002;";
            diagBtn.onclick = () => {
                tracer.active = !tracer.active;
                diagPanel.style.display = tracer.active ? "block" : "none";
                diagBtn.style.background = tracer.active ? "rgba(0,100,0,0.8)" : "rgba(0,0,0,0.5)";
            };
            canvasContainer.appendChild(diagBtn);

            const refreshTransforms = () => {
                try {
                    if (!currentModel) return;
                    const obj = loadedModels.find(m => m.model === currentModel);
                    if (!obj) return;

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
                    // Optimization: only dirty the graph if NOT currently animating/looping
                    // This prevents double-rendering during interaction
                    if (app.graph && !isAnimating) app.graph.setDirtyCanvas(true);
                } catch (e) {
                    console.error("[HY-Motion] refreshTransforms Error:", e);
                }
            };

            let _uiSyncTimeout = null;
            const throttledRefreshTransforms = () => {
                if (_uiSyncTimeout) return;
                _uiSyncTimeout = setTimeout(() => {
                    refreshTransforms();
                    _uiSyncTimeout = null;
                }, 150); // Increased throttle for LiteGraph
            };

            const throttledSyncWidgets = (model) => {
                const now = Date.now();
                if (node._lastWidgetSync && now - node._lastWidgetSync < 200) return;
                node._lastWidgetSync = now;
                syncWidgetsFromModel(model);
                // Force specialized dirty if not animating
                if (app.graph && !isAnimating) app.graph.setDirtyCanvas(true);
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

                    // if (app.graph) app.graph.setDirtyCanvas(true); // Moved to caller/throttle
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

                    input.onfocus = () => {
                        if (nodeData.name === "HYMotion3DModelLoader") {
                            // Only save once per focus session to avoid redundant states
                            if (!input._hasSavedUndo) {
                                saveStateForUndo();
                                input._hasSavedUndo = true;
                            }
                        }
                    };
                    input.onblur = () => {
                        input._hasSavedUndo = false;
                    };
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
                inPlaceBtn.style.display = "none";
                selectionInPlaceBtn.style.display = "none";
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

            // Hide transform widgets since they're controlled via the 3D canvas Transform panel
            if (nodeData.name === "HYMotion3DModelLoader") {
                const widgetsToHide = [
                    "translate_x", "translate_y", "translate_z",
                    "rotate_x", "rotate_y", "rotate_z",
                    "scale_x", "scale_y", "scale_z"
                ];
                setTimeout(() => {
                    if (this.widgets) {
                        this.widgets.forEach(w => {
                            if (widgetsToHide.includes(w.name)) {
                                w.type = "hidden";
                                if (w.element) w.element.style.display = "none";
                            }
                        });
                        // Trigger node resize to recalculate layout
                        if (app.graph) app.graph.setDirtyCanvas(true);
                    }
                }, 0);
            }


            let THREE = window.__HY_MOTION_THREE__ || null;
            let renderer, scene, camera, orbitControls, transformControl, clock;
            let currentModel = null, mixer = null;
            let skeletalSamples = [];
            let isPlaying = false;
            let currentFrame = 0;
            let maxFrames = 0;
            let startFrame = 0;
            let endFrame = 0;
            let frameAccumulator = 0;
            const targetFPS = 30;
            const frameTime = 1 / targetFPS;
            let mixers = [];
            let isInitialized = false;
            let pendingDataQueue = [];
            let lastMotionsData = null;
            let lastFbxUrl = null;
            let lastModelUrl = null;
            let lastTimestamp = 0;
            let animationFrameId = null;
            let isPoseMode = false;
            let selectedBone = null;
            let boneGizmos = [];
            let boneLines = [];
            let pivotIndicator = null;
            let restPoseData = [];
            let boneGizmoSize = 0.03;

            // Performance optimization flags
            let needsRender = true; // Dirty flag for smart rendering
            let isAnimating = false; // Track if animation loop should run
            let cachedCanvasRect = null; // Cache for getBoundingClientRect to prevent layout thrashing


            let raycaster = null;
            let selectedModels = []; // Array to support multi-selection
            let loadedModels = [];
            let isInPlace = false;

            // Undo/Redo stack for transforms
            const undoStack = [];
            const redoStack = [];
            const MAX_UNDO_STEPS = 50;

            // Save current transform state for undo
            const saveStateForUndo = () => {
                const state = loadedModels.map(m => {
                    const modelState = {
                        modelPath: m.modelPath,
                        position: m.model.position.clone(),
                        rotation: m.model.rotation.clone(),
                        scale: m.model.scale.clone(),
                        bones: {}
                    };

                    // Capture bone poses if in Pose Mode or if it's the Model Loader
                    if (isPoseMode || nodeData.name === "HYMotion3DModelLoader") {
                        m.model.traverse(obj => {
                            if (obj.isBone) {
                                modelState.bones[obj.name] = obj.quaternion.clone();
                            }
                        });
                    }
                    return modelState;
                });
                undoStack.push(state);
                if (undoStack.length > MAX_UNDO_STEPS) undoStack.shift();
                redoStack.length = 0; // Clear redo on new action
            };

            // Undo last transform
            const undo = () => {
                if (undoStack.length === 0) {
                    statusLabel.innerText = "Nothing to undo";
                    return;
                }
                // Save current state for redo
                const currentState = loadedModels.map(m => {
                    const modelState = {
                        modelPath: m.modelPath,
                        position: m.model.position.clone(),
                        rotation: m.model.rotation.clone(),
                        scale: m.model.scale.clone(),
                        bones: {}
                    };
                    m.model.traverse(obj => {
                        if (obj.isBone) {
                            modelState.bones[obj.name] = obj.quaternion.clone();
                        }
                    });
                    return modelState;
                });
                redoStack.push(currentState);

                const prevState = undoStack.pop();
                prevState.forEach(s => {
                    const m = loadedModels.find(lm => lm.modelPath === s.modelPath);
                    if (m && m.model) {
                        m.model.position.copy(s.position);
                        m.model.rotation.copy(s.rotation);
                        m.model.scale.copy(s.scale);

                        // Restore bone poses
                        if (s.bones) {
                            m.model.traverse(obj => {
                                if (obj.isBone && s.bones[obj.name]) {
                                    obj.quaternion.copy(s.bones[obj.name]);
                                }
                            });
                        }
                    }
                });
                syncWidgetsFromModel(currentModel);
                updateBoneGizmos();
                requestRender();
                statusLabel.innerText = "Undo";
            };

            // Redo last undone transform
            const redo = () => {
                if (redoStack.length === 0) {
                    statusLabel.innerText = "Nothing to redo";
                    return;
                }
                // Save current state for undo
                const currentState = loadedModels.map(m => {
                    const modelState = {
                        modelPath: m.modelPath,
                        position: m.model.position.clone(),
                        rotation: m.model.rotation.clone(),
                        scale: m.model.scale.clone(),
                        bones: {}
                    };
                    m.model.traverse(obj => {
                        if (obj.isBone) {
                            modelState.bones[obj.name] = obj.quaternion.clone();
                        }
                    });
                    return modelState;
                });
                undoStack.push(currentState);

                const nextState = redoStack.pop();
                nextState.forEach(s => {
                    const m = loadedModels.find(lm => lm.modelPath === s.modelPath);
                    if (m && m.model) {
                        m.model.position.copy(s.position);
                        m.model.rotation.copy(s.rotation);
                        m.model.scale.copy(s.scale);

                        // Restore bone poses
                        if (s.bones) {
                            m.model.traverse(obj => {
                                if (obj.isBone && s.bones[obj.name]) {
                                    obj.quaternion.copy(s.bones[obj.name]);
                                }
                            });
                        }
                    }
                });
                syncWidgetsFromModel(currentModel);
                updateBoneGizmos();
                requestRender();
                statusLabel.innerText = "Redo";
            };


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

            // Gizmo mode switching function (Shared Scope)
            let currentGizmoMode = 'none';
            const setGizmoMode = (mode) => {
                if (currentGizmoMode === mode) return;
                console.log(`[HY-Motion] Gizmo Mode: ${mode.toUpperCase()}`);
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
                if (transformControl) {
                    if (mode === 'none') {
                        transformControl.detach();
                    } else {
                        transformControl.setMode(mode);
                        // Attach to first selected model if any
                        if (selectedModels.length > 0 && selectedModels[0].type === 'model') {
                            transformControl.attach(selectedModels[0].obj.model);
                        }
                    }
                }
            };

            const initThree = async () => {
                if (isInitialized) return;

                try {
                    // Global Library Loading (Singleton)
                    if (!window.__HY_MOTION_THREE__) {
                        if (!isGlobalInitializing) {
                            isGlobalInitializing = true;
                            console.log("[HY-Motion] Loading 3D Engine Bundle...");
                            globalInitPromise = (async () => {
                                try {
                                    const [three, orbit, exporter] = await Promise.all([
                                        import(THREE_URL),
                                        import(ORBIT_CONTROLS_URL),
                                        import(GLTF_EXPORTER_URL)
                                    ]);
                                    window.__HY_MOTION_THREE__ = three;
                                    window.__HY_MOTION_ORBIT__ = orbit.OrbitControls;
                                    window.__HY_MOTION_EXPORTER__ = exporter.GLTFExporter;
                                } finally {
                                    isGlobalInitializing = false;
                                }
                            })();
                        }
                        await globalInitPromise;
                    }

                    if (!window.__HY_MOTION_THREE__) throw new Error("Failed to load 3D engine");

                    THREE = window.__HY_MOTION_THREE__;
                    const OrbitControls = window.__HY_MOTION_ORBIT__;
                    const GLTFExporter = window.__HY_MOTION_EXPORTER__;

                    if (!scene) {
                        scene = new THREE.Scene();
                        scene.background = new THREE.Color(0x111111);
                        console.log(`[HY-Motion ${node.id}] Scene created.`);
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

                        // Initialize cached rect to prevent layout thrashing on first pointer event
                        cachedCanvasRect = renderer.domElement.getBoundingClientRect();

                        // Store references on canvas for onResize callback access
                        renderer.domElement.__renderer = renderer;
                        renderer.domElement.__camera = camera;

                        canvasContainer.appendChild(renderer.domElement);

                        transformControl = new TransformControls(camera, renderer.domElement);
                        node.transformControl = transformControl; // EXPOSE TO NODE
                        scene.add(transformControl);

                        orbitControls = new OrbitControls(camera, renderer.domElement);
                        orbitControls.enableDamping = true;
                        orbitControls.dampingFactor = 0.05;
                        orbitControls.target.set(0, 0.8, 0);

                        // Trigger rendering when user interacts with camera
                        orbitControls.addEventListener('start', () => {
                            startAnimating(); // Start loop when user begins interaction
                        });
                        orbitControls.addEventListener('change', () => {
                            requestRender('camera-move');
                        });

                        // Allow middle mouse to rotate (like Blender)
                        orbitControls.mouseButtons = {
                            LEFT: THREE.MOUSE.ROTATE,
                            MIDDLE: THREE.MOUSE.ROTATE,
                            RIGHT: THREE.MOUSE.PAN
                        };

                        // Prevent orbit controls from interfering with gizmo
                        transformControl.addEventListener('dragging-changed', (event) => {
                            orbitControls.enabled = !event.value;
                            if (event.value) {
                                // Find best name for the object being dragged
                                let target = transformControl.object;
                                let objName = "unnamed";
                                if (target) {
                                    // Robust name search: check node name and user-facing path
                                    let root = target;
                                    while (root && !root.userData.modelPath && root.parent) root = root.parent;
                                    const modelPrefix = (root && root.userData.modelPath) ? (root.userData.modelPath.split('/').pop() + " > ") : "";
                                    objName = modelPrefix + (target.name || "unnamed");
                                }
                                console.log(`[HY-Motion] Drag START: ${objName} (${transformControl.mode})`);
                                startAnimating(); // Start loop when user begins interaction
                                saveStateForUndo(); // Save state for undo before transform
                            } else {
                                console.log(`[HY-Motion] Drag END`);
                            }
                        });

                        // Robustness: ensure orbit controls are re-enabled even if drag event is lost
                        window.addEventListener('mouseup', () => {
                            if (orbitControls && !orbitControls.enabled && (!transformControl.dragging)) {
                                orbitControls.enabled = true;
                                requestRender();
                            }
                        }, { passive: true });
                        transformControl.addEventListener('change', () => {
                            if (transformControl.object && currentModel === transformControl.object) {
                                throttledSyncWidgets(currentModel);
                            }
                            requestRender();
                        });


                        // Button click handlers
                        translateBtn.onclick = () => setGizmoMode('translate');
                        rotateBtn.onclick = () => setGizmoMode('rotate');
                        scaleBtn.onclick = () => setGizmoMode('scale');
                        gizmoOffBtn.onclick = () => setGizmoMode('none');

                        // Keyboard shortcuts (Blender-style) - only when canvas/container is focused or hovered
                        const handleKeyPress = (e) => {
                            // RELIABLE HOVER CHECK: Check actual DOM state at keypress time
                            // instead of relying on mouseenter/leave events which can be stale
                            const isCanvasFocused = document.activeElement === renderer.domElement;
                            const isOverContainer = canvasContainer.matches(':hover');
                            const shouldIntercept = isCanvasFocused || isOverContainer;

                            // Log all undo/redo attempts for debugging
                            if ((e.key.toLowerCase() === 'z' || e.key.toLowerCase() === 'y') && (e.ctrlKey || e.metaKey)) {
                                console.log(`[HY-Motion Node ${node.id}] Undo/Redo key=${e.key}, focused=${isCanvasFocused}, hover=${isOverContainer}, intercept=${shouldIntercept}`);
                            }

                            // Only handle shortcuts when canvas is focused or mouse is over container
                            if (!shouldIntercept) {
                                // Allow ComfyUI to handle the event when not focused/hovering
                                return;
                            }

                            // Disable gizmo shortcuts for Legacy Player
                            if (nodeData.name === "HYMotionFBXPlayer") return;
                            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                            switch (e.key.toLowerCase()) {
                                case 'g':
                                    console.log("[HY-Motion] Keyboard: Translate (G)");
                                    setGizmoMode('translate');
                                    e.preventDefault();
                                    e.stopPropagation();
                                    e.stopImmediatePropagation();
                                    break;
                                case 'r':
                                    console.log("[HY-Motion] Keyboard: Rotate (R)");
                                    setGizmoMode('rotate');
                                    e.preventDefault();
                                    e.stopPropagation();
                                    e.stopImmediatePropagation();
                                    break;
                                case 's':
                                    console.log("[HY-Motion] Keyboard: Scale (S)");
                                    setGizmoMode('scale');
                                    e.preventDefault();
                                    e.stopPropagation();
                                    e.stopImmediatePropagation();
                                    break;
                                case 'escape':
                                    console.log("[HY-Motion] Keyboard: Disable Gizmo (Esc)");
                                    setGizmoMode('none');
                                    e.preventDefault();
                                    e.stopPropagation();
                                    e.stopImmediatePropagation();
                                    break;
                                case 'z':
                                    // Use Alt+Z for viewer undo to avoid conflict with ComfyUI's Ctrl+Z
                                    if (e.altKey && !e.ctrlKey && !e.metaKey) {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        e.stopImmediatePropagation();

                                        if (e.shiftKey) {
                                            console.log(`[HY-Motion Node ${node.id}] Viewer Redo (Alt+Shift+Z)`);
                                            redo();
                                        } else {
                                            console.log(`[HY-Motion Node ${node.id}] Viewer Undo (Alt+Z)`);
                                            undo();
                                        }
                                        return false;
                                    }
                                    break;
                                case 'y':
                                    // Use Alt+Y for viewer redo
                                    if (e.altKey && !e.ctrlKey && !e.metaKey) {
                                        console.log(`[HY-Motion Node ${node.id}] Viewer Redo (Alt+Y)`);
                                        redo();
                                        e.preventDefault();
                                        e.stopPropagation();
                                        e.stopImmediatePropagation();
                                        return false;
                                    }
                                    break;
                            }
                        };

                        // Use capture phase to intercept events before ComfyUI sees them
                        document.addEventListener('keydown', handleKeyPress, true);

                        // Store for later use
                        node.transformControl = transformControl;
                        node.setGizmoMode = setGizmoMode;

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
                                    // Cache the rect to prevent layout thrashing in pointer events
                                    cachedCanvasRect = renderer.domElement.getBoundingClientRect();
                                    requestRender(); // Trigger single render after resize
                                }
                            }, 200);
                        });
                        resizeObserver.observe(canvasContainer);

                        // Initialize hit proxies and highlights whenever we are rendering to ensure consistency
                        raycaster = new THREE.Raycaster();

                        // Optimized raycasting with throttling
                        let lastRaycastTime = 0;
                        const raycastInterval = 100; // 10fps for hover detection

                        const throttledPointerMove = (event) => {
                            const now = Date.now();
                            // Even more aggressive throttling: 5fps for hover, 30fps if clicking
                            const interval = (event.buttons > 0) ? 33 : 200;
                            if (now - lastRaycastTime < interval) return;
                            lastRaycastTime = now;
                            onCanvasPointerMove(event);
                        };

                        canvasContainer.addEventListener('pointerdown', (e) => {
                            // Focus the canvas to capture keyboard events reliably
                            renderer.domElement.focus();
                            onCanvasPointerDown(e);
                        });
                        canvasContainer.addEventListener('pointermove', (e) => {
                            throttledPointerMove(e);
                            // Ensure render loop wakes up on hover if not already running
                            if (!isAnimating) requestRender();
                        });

                        // Intersection Observer to pause rendering when not visible
                        let _visibilityTimeout = null;
                        const visibilityObserver = new IntersectionObserver((entries) => {
                            entries.forEach(entry => {
                                if (_visibilityTimeout) clearTimeout(_visibilityTimeout);
                                _visibilityTimeout = setTimeout(() => {
                                    if (entry.isIntersecting) {
                                        startAnimating("visible");
                                    } else {
                                        stopAnimating("off-screen");
                                    }
                                }, 150);
                            });
                        }, { threshold: 0.1 });
                        visibilityObserver.observe(container);

                        // Store observers for cleanup
                        node._visibilityObserver = visibilityObserver;
                        node._resizeObserver = resizeObserver;
                        node._handleKeyPress = handleKeyPress;
                    }


                    isInitialized = true;
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

                // Add bone gizmos first (highest priority in pose mode)
                if (isPoseMode && boneGizmos.length > 0) {
                    intersectableObjects.push(...boneGizmos);
                }

                for (const obj of loadedModels) {
                    if (obj.hitProxy) intersectableObjects.push(obj.hitProxy);
                    if (isPoseMode && obj.model) {
                        // Add the model itself for bone raycasting fallback
                        intersectableObjects.push(obj.model);
                    }
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
                // Use cached rect to prevent layout thrashing
                const rect = cachedCanvasRect || renderer.domElement.getBoundingClientRect();
                const mouse = new THREE.Vector2();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);

                // Check intersects using proxies/joints
                const intersects = raycaster.intersectObjects(getIntersectables(), true);
                renderer.domElement.style.cursor = intersects.length > 0 ? "pointer" : "auto";
            };

            const onCanvasPointerDown = (event) => {
                if (!raycaster || !camera || !scene) return;
                console.log("[HY-Motion] onCanvasPointerDown fired via", event.target.tagName);

                // Prevent interfering with orbit controls mouse/touch (Left-click only for selection)
                if (event.button !== 0) return;

                // Priority 0: If hovering/clicking Transform gizmo, skip selection logic
                if (transformControl && (transformControl.axis !== null || transformControl.dragging)) return;

                // Use cached rect to prevent layout thrashing
                // FORCE FRESH RECT: Caching here causes flaky selection if the page scrolls/layouts shift
                const rect = renderer.domElement.getBoundingClientRect();
                const mouse = new THREE.Vector2();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersectables = getIntersectables();
                console.log(`[HY-Motion] Click - Intersectable objects: ${intersectables.length}`, {
                    boneGizmos: boneGizmos.length,
                    hitProxies: loadedModels.filter(m => m.hitProxy).length,
                    isPoseMode
                });

                const intersects = raycaster.intersectObjects(intersectables, true);
                console.log(`[HY-Motion] Intersections found: ${intersects.length}`,
                    intersects.map(i => ({
                        type: i.object.userData.type || 'unknown',
                        name: i.object.name || 'unnamed',
                        isBoneGizmo: i.object.userData.type === 'bone_gizmo',
                        hasSelectableParent: !!i.object.userData.selectableParent
                    }))
                );

                if (intersects.length > 0) {
                    // Start interaction - ensure loop is active
                    startAnimating();

                    // Pose Mode: Try to find a bone first
                    if (isPoseMode) {
                        // Prioritize bone gizmos (spheres)
                        let boneHit = null;
                        for (let i = 0; i < intersects.length; i++) {
                            if (intersects[i].object.userData.type === "bone_gizmo") {
                                boneHit = intersects[i].object.userData.bone;
                                console.log(`[HY-Motion] ✓ Bone GIZMO clicked: ${boneHit.name}`);
                                break;
                            }
                        }

                        if (boneHit) {
                            selectBone(boneHit);
                            event.stopPropagation();
                            return;
                        } else {
                            console.log(`[HY-Motion] ✗ No bone gizmo hit in pose mode`);
                        }

                        // Fallback to raycasting the model itself
                        let bone = null;
                        for (let i = 0; i < intersects.length; i++) {
                            let obj = intersects[i].object;
                            while (obj) {
                                if (obj.isBone) {
                                    bone = obj;
                                    break;
                                }
                                obj = obj.parent;
                            }
                            if (bone) break;
                        }
                        if (bone) {
                            console.log(`[HY-Motion] ✓ Bone clicked (mesh fallback): ${bone.name}`);
                            selectBone(bone);
                            event.stopPropagation();
                            return;
                        } else {
                            console.log(`[HY-Motion] ✗ No bone found via mesh in pose mode`);
                        }
                    }

                    // Find first valid hit
                    let hit = null;
                    for (let i = 0; i < intersects.length; i++) {
                        if (intersects[i].object.userData.selectableParent) {
                            hit = intersects[i].object;
                            console.log(`[HY-Motion] ✓ HIT PROXY clicked for model: ${hit.userData.selectableParent.name}`);
                            break;
                        }
                    }

                    if (hit) {
                        selectObject(hit.userData.selectableParent, hit.userData.type, event.ctrlKey);
                        event.stopPropagation();
                        // DO NOT preventDefault here as it might break OrbitControls
                    } else {
                        console.log(`[HY-Motion] ✗ No selectable object found (no hit proxy matched)`);
                    }
                } else {
                    // Debug logging for missed clicks
                    console.log(`[HY-Motion] Missed click. Mouse: ${mouse.x.toFixed(3)}, ${mouse.y.toFixed(3)}`);
                    console.log(`[HY-Motion] Canvas Rect: W=${rect.width}, H=${rect.height}, T=${rect.top}, L=${rect.left}`);

                    console.log(`[HY-Motion] Background click - deselecting${!event.ctrlKey ? ' all' : ' (Ctrl held)'}`);
                    // Clicked on background
                    if (!event.ctrlKey) {
                        deselectAll();
                    }
                }
            };

            const selectBone = (bone) => {
                // Ensure we don't have residual model selections
                deselectAll();

                selectedBone = bone;

                if (transformControl) {
                    transformControl.attach(bone);
                    // For bones, we usually only want rotation
                    transformControl.setMode('rotate');
                    // Update gizmo buttons to reflect mode via shared scope function
                    setGizmoMode('rotate');
                    requestRender("select-bone");
                }

                updatePivotIndicator();
                updateBoneGizmos(); // Highlight selected gizmo

                statusLabel.innerText = `Selected Bone: ${bone.name}`;
                statusLabel.style.color = "#0ff";
                requestRender();
            };

            const serializePose = () => {
                if (!currentModel) return "{}";
                const pose = { bones: {}, root_pos: [0, 0, 0] };
                currentModel.traverse(obj => {
                    if (obj.isBone) {
                        pose.bones[obj.name] = {
                            rot: [obj.quaternion.x, obj.quaternion.y, obj.quaternion.z, obj.quaternion.w],
                            pos: [obj.position.x, obj.position.y, obj.position.z]
                        };
                    }
                });

                const rootBone = findRootBone(currentModel);
                if (rootBone) {
                    // Combine model position (global gizmo) and root bone position (bone gizmo)
                    // This captures the total viewport location of the character.
                    pose.root_pos = [
                        currentModel.position.x + rootBone.position.x,
                        currentModel.position.y + rootBone.position.y,
                        currentModel.position.z + rootBone.position.z
                    ];
                }

                return JSON.stringify(pose);
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

            // Pre-allocated vectors for GC optimization (initialized lazily)
            let _vec3_1 = null;
            let _vec3_2 = null;
            let _box_1 = null;

            const getRealtimeBox = (obj) => {
                if (!_vec3_1) _vec3_1 = new THREE.Vector3();
                if (!_box_1) _box_1 = new THREE.Box3();

                const box = new THREE.Box3();
                // Cache meshes/bones on the object to avoid traverse every frame
                if (!obj._internal_mesh_cache) {
                    obj._internal_mesh_cache = [];
                    obj.traverse(child => {
                        if (child.isMesh) obj._internal_mesh_cache.push(child);
                    });
                }

                obj._internal_mesh_cache.forEach(child => {
                    if (child.isSkinnedMesh && child.skeleton) {
                        _box_1.makeEmpty();
                        const bones = child.skeleton.bones;
                        for (let i = 0; i < bones.length; i++) {
                            // PERFORMANCE: Use matrixWorld directly, assumed updated by caller
                            _vec3_1.setFromMatrixPosition(bones[i].matrixWorld);
                            _box_1.expandByPoint(_vec3_1);
                        }
                        _box_1.expandByScalar(0.15); // Add margin for joints
                        box.union(_box_1);
                    } else {
                        // For non-skinned, update once and use bounding box
                        child.updateMatrixWorld(false);
                        _box_1.setFromObject(child);
                        if (!_box_1.isEmpty()) box.union(_box_1);
                    }
                });
                return box;
            };

            const selectObject = (obj, type, isMulti = false) => {
                if (obj) console.log(`[HY-Motion] Select ${type.toUpperCase()}: ${obj.name || "unnamed"}`);
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
                if (nodeData.name !== "HYMotionFBXPlayer" && transformControl && type === 'model') {
                    const currentMode = [translateBtn, rotateBtn, scaleBtn].find(btn =>
                        btn.style.background === 'rgb(0, 102, 204)' || btn.style.background === '#0066cc'
                    )?.dataset.mode;

                    if (currentMode && currentMode !== 'none') {
                        transformControl.attach(obj.model);
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

                // Update In-Place Button state based on selection
                if (count > 0 && nodeData.name !== "HYMotion3DModelLoader") {
                    // Has selection: show per-selection in-place state
                    const allInPlace = selectedModels.every(s => s.obj.isInPlace);
                    inPlaceBtn.innerText = allInPlace ? "🧍‍♂️" : "🏃‍♂️";
                    inPlaceBtn.style.background = allInPlace ? "#0066cc" : "#444";
                    inPlaceBtn.title = `In-Place: ${count} selected (${allInPlace ? 'ON' : 'OFF'})`;
                } else {
                    // No selection: show global in-place state
                    inPlaceBtn.innerText = isInPlace ? "🧍‍♂️" : "🏃‍♂️";
                    inPlaceBtn.style.background = isInPlace ? "#0066cc" : "#444";
                    inPlaceBtn.title = `Global In-Place: ${isInPlace ? 'ON' : 'OFF'}`;
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
                if (selectedModels.length > 0) console.log("[HY-Motion] Deselect All");
                while (selectedModels.length > 0) {
                    removeSelectionAt(0);
                }
                selectedBone = null;
                // Detach gizmo when deselecting bones
                if (transformControl && !transformControl.dragging) {
                    transformControl.detach();
                }
                updateUI();
                updateBoneGizmos(); // Update visual feedback
                requestRender();
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
                        const modelIsInPlace = selection.obj.isInPlace || isInPlace;

                        // Build the file path
                        let filePath = fileType === 'output' ? `output/${subfolder ? subfolder + '/' : ''}${fileName}` : fileName;

                        if (modelIsInPlace) {
                            // Use the in-place export endpoint to bake in-place into the FBX
                            console.log(`[HY-Motion] Exporting with in-place: ${fileName}`);
                            try {
                                const response = await api.fetchApi("/hymotion/export_inplace", {
                                    method: "POST",
                                    body: JSON.stringify({ input_path: filePath })
                                });

                                if (response.ok) {
                                    const blob = await response.blob();
                                    const link = document.createElement('a');
                                    link.href = URL.createObjectURL(blob);
                                    link.download = fileName.replace(".fbx", "_inplace.fbx");
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                    URL.revokeObjectURL(link.href);
                                } else {
                                    const errData = await response.json();
                                    console.error("[HY-Motion] In-place export failed:", errData.error);
                                    alert(`In-place export failed: ${errData.error}\nFalling back to normal download.`);
                                    // Fallback to normal download
                                    let fetchUrl = `${window.location.origin}/view?type=${encodeURIComponent(fileType || 'output')}&filename=${encodeURIComponent(fileName)}`;
                                    if (subfolder) fetchUrl += `&subfolder=${encodeURIComponent(subfolder)}`;
                                    const link = document.createElement('a');
                                    link.href = fetchUrl;
                                    link.download = fileName;
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                }
                            } catch (e) {
                                console.error("[HY-Motion] In-place export error:", e);
                                alert(`In-place export error: ${e.message}`);
                            }
                        } else {
                            // Normal download without in-place
                            let fetchUrl = `${window.location.origin}/view?type=${encodeURIComponent(fileType || 'output')}&filename=${encodeURIComponent(fileName)}`;
                            if (subfolder) fetchUrl += `&subfolder=${encodeURIComponent(subfolder)}`;

                            const link = document.createElement('a');
                            link.href = fetchUrl;
                            link.download = fileName;
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                        }

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

            let _warmLoopTimeout = null;
            // Helper to start animation loop
            const startAnimating = (reason = "unknown") => {
                const isReady = isInitialized && renderer && scene;
                if (!isReady) {
                    return;
                }

                if (_warmLoopTimeout) {
                    clearTimeout(_warmLoopTimeout);
                    _warmLoopTimeout = null;
                }
                // Robustness: start if not animating OR if loop died (no frame ID)
                if (!isAnimating || !animationFrameId) {
                    console.log(`[HY-Motion ${node.id}] START animation loop (reason: ${reason}, poseMode: ${isPoseMode})`);
                    isAnimating = true;
                    if (!animationFrameId) animate();
                }
            };

            // Helper to stop animation loop
            const stopAnimating = (reason = "unknown") => {
                if (!isAnimating) return;

                // Persistence: stay "warm" for 500ms to avoid staccato restarts during interaction gaps
                if (reason === "idle") {
                    if (!_warmLoopTimeout) {
                        _warmLoopTimeout = setTimeout(() => {
                            _warmLoopTimeout = null;
                            stopAnimating("warm-timeout");
                        }, 500);
                    }
                    return; // Hold the loop open
                }

                // Actually kill the loop
                console.log(`[HY-Motion ${node.id}] STOP animation loop (reason: ${reason})`);
                isAnimating = false;
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }
            };

            // Helper to request a single render
            const requestRender = (reason = "request") => {
                needsRender = true;
                // Interaction clears the warm-down timeout immediately
                if (_warmLoopTimeout) {
                    clearTimeout(_warmLoopTimeout);
                    _warmLoopTimeout = null;
                }
                if (!isAnimating) startAnimating(reason);
            };

            const animate = () => {
                try {
                    if (!isAnimating || !isInitialized || !renderer) {
                        if (isAnimating) {
                            console.warn(`[HY-Motion ${node.id}] animate() bailed: isAnimating=${isAnimating}, isInit=${isInitialized}, renderer=${!!renderer}`);
                            isAnimating = false;
                        }
                        return;
                    }

                    // Guard against double scheduling
                    if (animationFrameId) cancelAnimationFrame(animationFrameId);

                    // If THREE is missing, try fallback before dropping to idle
                    if (!THREE) {
                        THREE = window.__HY_MOTION_THREE__;
                        if (!THREE) {
                            console.warn(`[HY-Motion ${node.id}] THREE is null - stopping animation loop`);
                            isAnimating = false;
                            animationFrameId = null;
                            return;
                        }
                    }

                    animationFrameId = requestAnimationFrame(animate);

                    const frameStart = performance.now();
                    const delta = clock ? clock.getDelta() : 0.016;
                    tracer.mark("frame-start");

                    // Slow Frame Detection: skip expensive tasks if prev frame was > 200ms
                    const isHeavyLoad = node._lastFrameTime > 200;

                    tracer.mark("logic-start");
                    // OrbitControls Damping Update
                    if (orbitControls) {
                        tracer.mark("controls-start");
                        const controlsChanged = orbitControls.update();
                        tracer.mark("controls-end");
                        tracer.measure("controls-dur", "controls-start", "controls-end");

                        if (controlsChanged) needsRender = true;
                    }

                    let shouldRender = needsRender;

                    if (isPlaying && maxFrames > 0 && !isScrubbing) {
                        shouldRender = true;
                        frameAccumulator += delta;

                        const sTime = startFrame * frameTime;
                        const eTime = (endFrame > 0) ? (endFrame * frameTime) : (maxFrames * frameTime);
                        const range = Math.max(0.001, eTime - sTime);

                        if (frameAccumulator < sTime) frameAccumulator = sTime;
                        if (frameAccumulator >= eTime) {
                            frameAccumulator = sTime + (frameAccumulator - sTime) % range;
                        }

                        currentFrame = frameAccumulator / frameTime;

                        if (mixers.length > 0) {
                            const loopTime = maxFrames * frameTime;
                            for (const m of mixers) {
                                m.setTime(frameAccumulator % loopTime);
                            }
                        }

                        updateSkeletons(currentFrame);

                        // FBX In-Place Logic
                        if (mixers.length > 0) {
                            for (const m of mixers) {
                                const modelObj = loadedModels.find(obj => obj.model === m.getRoot());
                                const activeInPlace = isInPlace || (modelObj && modelObj.isInPlace);

                                if (activeInPlace) {
                                    const root = modelObj._cached_root || findRootBone(m.getRoot());
                                    if (root) {
                                        modelObj._cached_root = root;
                                        root.position.x = 0;
                                        root.position.z = 0;
                                    }
                                }
                            }
                        }

                        if (!isScrubbing && Math.floor(currentFrame) % 10 === 0) {
                            const nextVal = (currentFrame / maxFrames) * 100;
                            if (Math.abs(progress.value - nextVal) > 0.5) {
                                progress.value = nextVal;
                            }
                        }
                    }
                    tracer.mark("logic-end");
                    tracer.measure("logic-dur", "logic-start", "logic-end");

                    // In pose mode, always render to keep bone gizmos smooth
                    if (isPoseMode) shouldRender = true;

                    if (shouldRender && renderer && scene && camera) {
                        const now = Date.now();
                        const isModelLoader = nodeData.name === "HYMotion3DModelLoader";
                        const isInteracting = transformControl && transformControl.dragging;

                        // PERFORMANCE: No throttling during interaction, playback, or pose mode
                        const throttleTime = (isModelLoader || isInteracting || isPlaying || isPoseMode) ? 0 : 100;

                        if (!node._lastHeavyUpdate || now - node._lastHeavyUpdate >= throttleTime) {
                            const bgThrottle = (isModelLoader || isPlaying || needsRender || isInteracting || isPoseMode) ? 0 : 500;

                            // Expensive task mitigation - but ALWAYS run in pose mode for smooth bone dragging
                            if ((!isHeavyLoad || isPoseMode) && (!node._lastBgTaskUpdate || now - node._lastBgTaskUpdate >= bgThrottle)) {
                                tracer.mark("bg-start");
                                updateHitProxies();
                                updateSelectionHighlights();
                                updateBoneGizmos();
                                node._lastBgTaskUpdate = now;
                                tracer.mark("bg-end");
                                tracer.measure("bg-dur", "bg-start", "bg-end");
                            }
                            node._lastHeavyUpdate = now;
                        }

                        tracer.mark("render-start");
                        renderer.render(scene, camera);
                        tracer.mark("render-end");
                        tracer.measure("render-dur", "render-start", "render-end");

                        needsRender = false;
                    }

                    tracer.report();
                    node._lastFrameTime = performance.now() - frameStart;

                    // Intelligent Sleep: Keep loop running in pose mode for smooth bone gizmo updates
                    if (!isPlaying && !needsRender && !isPoseMode) {
                        stopAnimating("idle");
                    }
                } catch (err) {
                    console.error(`[HY-Motion ${node.id}] Animation loop error:`, err);
                    // Don't stop the loop for non-fatal errors, but log them
                }
            };

            const createBoneGizmos = (model) => {
                // Clear existing and properly dispose to free GPU memory
                boneGizmos.forEach(g => {
                    scene.remove(g);
                    if (g.geometry) g.geometry.dispose();
                    if (g.material) g.material.dispose();
                });
                boneGizmos = [];
                restPoseData = [];

                if (!model) return;

                // Shared geometry and material to reduce draw call overhead
                const gizmoGeo = new THREE.SphereGeometry(boneGizmoSize, 12, 12);

                model.traverse((child) => {
                    if (child.isBone) {
                        const gizmo = new THREE.Mesh(
                            gizmoGeo,
                            new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: xraySkeleton })
                        );
                        gizmo.userData.bone = child;
                        gizmo.userData.type = "bone_gizmo";
                        scene.add(gizmo);
                        boneGizmos.push(gizmo);

                        // Store rest pose
                        restPoseData.push({
                            bone: child,
                            position: child.position.clone(),
                            rotation: child.rotation.clone(),
                            scale: child.scale.clone()
                        });
                    }
                });
                createBoneLines(model);
                updateBoneGizmos();
            };

            const createBoneLines = (model) => {
                boneLines.forEach(l => {
                    scene.remove(l);
                    if (l.geometry) l.geometry.dispose();
                    // materials are shared, don't dispose here
                });
                boneLines = [];

                if (!model) return;

                const lineMat = new THREE.LineBasicMaterial({ color: 0x888888, depthTest: xraySkeleton, transparent: true, opacity: 0.5 });

                model.traverse((child) => {
                    if (child.isBone && child.parent && child.parent.isBone) {
                        const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
                        const line = new THREE.Line(geo, lineMat);
                        line.userData.bone = child;
                        line.userData.parentBone = child.parent;
                        scene.add(line);
                        boneLines.push(line);
                    }
                });
            };

            const updateBoneGizmos = () => {
                if (!isPoseMode || !showSkeleton || !currentModel) {
                    boneGizmos.forEach(g => g.visible = false);
                    boneLines.forEach(l => l.visible = false);
                    if (pivotIndicator) pivotIndicator.visible = false;
                    return;
                }

                const now = Date.now();
                const isInteracting = transformControl && transformControl.dragging;
                const isAnimatingModel = isPlaying && maxFrames > 0;

                // PERFORMANCE FIX: Skip throttling entirely during bone dragging for smooth 60fps
                // Only throttle when idle to save resources
                if (!isInteracting && !isAnimatingModel) {
                    const minInterval = 100; // 10fps when idle
                    if (node._lastBoneGizmoUpdate && now - node._lastBoneGizmoUpdate < minInterval) {
                        return;
                    }
                }
                node._lastBoneGizmoUpdate = now;

                if (!_vec3_1) _vec3_1 = new THREE.Vector3();
                if (!_vec3_2) _vec3_2 = new THREE.Vector3();

                // CRITICAL PERFORMANCE: Update model matrix world ONCE per update block
                // This prevents getWorldPosition from re-traversing the entire hierarchy for every bone/line.
                currentModel.updateMatrixWorld(true);

                // OPTIMIZATION: When dragging, only update the selected bone and its descendants
                // This dramatically reduces computation from O(all bones) to O(affected bones)
                const bonesToUpdate = new Set();
                if (isInteracting && selectedBone) {
                    // Add selected bone
                    bonesToUpdate.add(selectedBone);
                    // Add all descendants
                    selectedBone.traverse(child => {
                        if (child.isBone) bonesToUpdate.add(child);
                    });
                } else {
                    // When not dragging, update all bones
                    boneGizmos.forEach(g => bonesToUpdate.add(g.userData.bone));
                }

                boneGizmos.forEach(g => {
                    const bone = g.userData.bone;
                    const shouldUpdate = bonesToUpdate.has(bone);

                    g.visible = true;

                    if (shouldUpdate) {
                        // Material property check to avoid redundant GPU state changes
                        if (g.material.depthTest !== xraySkeleton) g.material.depthTest = xraySkeleton;

                        // Directly access matrixWorld to avoid overhead
                        _vec3_1.setFromMatrixPosition(bone.matrixWorld);
                        g.position.copy(_vec3_1);
                    }

                    // Highlight selected (always check, even if not updating position)
                    if (selectedBone && bone === selectedBone) {
                        if (g.material.color.getHex() !== 0x00ffff) {
                            g.material.color.set(0x00ffff);
                            g.scale.setScalar(1.5);
                        }
                    } else {
                        if (g.material.color.getHex() !== 0xffffff) {
                            g.material.color.set(0xffffff);
                            g.scale.setScalar(1.0);
                        }
                    }
                });

                boneLines.forEach(l => {
                    l.visible = true;
                    const shouldUpdate = bonesToUpdate.has(l.userData.bone) || bonesToUpdate.has(l.userData.parentBone);

                    if (shouldUpdate) {
                        if (l.material.depthTest !== xraySkeleton) l.material.depthTest = xraySkeleton;

                        _vec3_1.setFromMatrixPosition(l.userData.bone.matrixWorld);
                        _vec3_2.setFromMatrixPosition(l.userData.parentBone.matrixWorld);

                        const posAttr = l.geometry.attributes.position;
                        posAttr.setXYZ(0, _vec3_1.x, _vec3_1.y, _vec3_1.z);
                        posAttr.setXYZ(1, _vec3_2.x, _vec3_2.y, _vec3_2.z);
                        posAttr.needsUpdate = true;
                    }
                });

                updatePivotIndicator();
            };

            const updatePivotIndicator = () => {
                if (!selectedBone || !isPoseMode || !showSkeleton) {
                    if (pivotIndicator) pivotIndicator.visible = false;
                    return;
                }

                if (!pivotIndicator) {
                    const group = new THREE.Group();
                    const ringGeo = new THREE.RingGeometry(0.08, 0.09, 32);
                    const matX = new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.DoubleSide, depthTest: false });
                    const matY = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.DoubleSide, depthTest: false });
                    const matZ = new THREE.MeshBasicMaterial({ color: 0x0000ff, side: THREE.DoubleSide, depthTest: false });

                    const ringX = new THREE.Mesh(ringGeo, matX);
                    ringX.rotation.y = Math.PI / 2;
                    const ringY = new THREE.Mesh(ringGeo, matY);
                    ringY.rotation.x = Math.PI / 2;
                    const ringZ = new THREE.Mesh(ringGeo, matZ);

                    group.add(ringX, ringY, ringZ);
                    scene.add(group);
                    pivotIndicator = group;
                }

                pivotIndicator.visible = true;
                const tempPos = new THREE.Vector3();
                selectedBone.getWorldPosition(tempPos);
                pivotIndicator.position.copy(tempPos);

                // Match bone rotation for local axes
                const tempQuat = new THREE.Quaternion();
                selectedBone.getWorldQuaternion(tempQuat);
                pivotIndicator.quaternion.copy(tempQuat);
            };

            const resetToRestPose = () => {
                restPoseData.forEach(data => {
                    data.bone.position.copy(data.position);
                    data.bone.rotation.copy(data.rotation);
                    data.bone.scale.copy(data.scale);
                });
                updateBoneGizmos();
                if (transformControl) transformControl.detach();
                selectedBone = null;
            };
            // Utility to properly dispose of Three.js objects
            const disposeObject = (obj) => {
                if (!obj) return;
                obj.traverse((child) => {
                    if (child.isMesh) {
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
                    }
                    // Specialized dispose for skeletal helpers
                    if (child.skeleton) child.skeleton.dispose();
                    if (child.isBone) { child._internal_mesh_cache = null; }
                });
                if (obj.dispose) obj.dispose();
                obj._internal_mesh_cache = null;
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
                if (!scene || !THREE) return; // Guard against race condition during init
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
                        console.log(`[HY-Motion] ✓ Created HIT BOX for model: ${obj.name || 'unnamed'}`);
                    }

                    // Only update at 10fps for performance
                    if (!obj._lastProxyUpdate || now - obj._lastProxyUpdate > 100) {
                        const box = getCachedBBox(obj.model);
                        if (!box.isEmpty()) {
                            const size = box.getSize(new THREE.Vector3());
                            const center = box.getCenter(new THREE.Vector3());
                            obj.hitProxy.scale.set(size.x * 1.03, size.y * 1.05, size.z * 1.05);
                            obj.hitProxy.position.copy(center);
                            obj._lastProxyUpdate = now;
                        }
                    }
                }
            };

            // Separate function to update selection highlights (only when needed)
            const updateSelectionHighlights = () => {
                if (!scene || !THREE) return; // Guard against race condition during init
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

                // Reset maxFrames if no skeletons are present to allow new models to set the duration
                if (skeletalSamples.length === 0) {
                    maxFrames = 0;
                }

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
                    const jointMat = new THREE.MeshStandardMaterial({ color: sampleColor, emissive: sampleColor, emissiveIntensity: 0.3, depthTest: false, transparent: true });
                    const boneMat = new THREE.LineBasicMaterial({ color: sampleColor, transparent: true, opacity: 0.8, depthTest: false });
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
                    skeletalSamples.push({ group, joints, bones, keypoints: m.keypoints, isInPlace: false });
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
                // Auto-centering disabled as per user request
                // if (loadedModels.length === 0) centerCameraOnObject(masterGroup);
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
                            let x = p1[0] + (p2[0] - p1[0]) * alpha;
                            let y = p1[1] + (p2[1] - p1[1]) * alpha;
                            let z = p1[2] + (p2[2] - p1[2]) * alpha;

                            if (isInPlace || sample.isInPlace) {
                                // Subtract root horizontal displacement
                                // Root is joint 0 in SMPL-H
                                const root1 = pos1[0];
                                const root2 = pos2[0];
                                const rootX = root1[0] + (root2[0] - root1[0]) * alpha;
                                const rootZ = root1[2] + (root2[2] - root1[2]) * alpha;

                                // We keep the initial root position to avoid jumping
                                const startRoot = sample.keypoints[0][0];
                                x -= (rootX - startRoot[0]);
                                z -= (rootZ - startRoot[2]);
                            }

                            sample.joints[i].position.set(x, y, z);
                        }
                    }

                    for (const b of sample.bones) {
                        const j1p1 = pos1[b.j1], j1p2 = pos1[b.j2];
                        const j2p1 = pos2[b.j1], j2p2 = pos2[b.j2];

                        if (j1p1 && j1p2 && j2p1 && j2p2 && THREE) {
                            if (!_vec3_1) _vec3_1 = new THREE.Vector3();
                            // Interpolate bone endpoints using pre-allocated vectors
                            _vec3_1.set(
                                j1p1[0] + (j2p1[0] - j1p1[0]) * alpha,
                                j1p1[1] + (j2p1[1] - j1p1[1]) * alpha,
                                j1p1[2] + (j2p1[2] - j1p1[2]) * alpha
                            );
                            if (!_vec3_2) _vec3_2 = new THREE.Vector3();
                            _vec3_2.set(
                                j1p2[0] + (j2p2[0] - j1p2[0]) * alpha,
                                j1p2[1] + (j2p2[1] - j1p2[1]) * alpha,
                                j1p2[2] + (j2p2[2] - j1p2[2]) * alpha
                            );

                            if (isInPlace || sample.isInPlace) {
                                const root1 = pos1[0];
                                const root2 = pos2[0];
                                const rootX = root1[0] + (root2[0] - root1[0]) * alpha;
                                const rootZ = root1[2] + (root2[2] - root1[2]) * alpha;
                                const startRoot = sample.keypoints[0][0];

                                const dx = rootX - startRoot[0];
                                const dz = rootZ - startRoot[2];
                                _vec3_1.x -= dx;
                                _vec3_1.z -= dz;
                                _vec3_2.x -= dx;
                                _vec3_2.z -= dz;
                            }

                            // Direct position buffer update for max performance
                            const posAttr = b.line.geometry.attributes.position;
                            posAttr.setXYZ(0, _vec3_1.x, _vec3_1.y, _vec3_1.z);
                            posAttr.setXYZ(1, _vec3_2.x, _vec3_2.y, _vec3_2.z);
                            posAttr.needsUpdate = true;
                        }
                    }
                }
            };

            const findRootBone = (object) => {
                if (!object) return null;
                let root = null;
                object.traverse((child) => {
                    if (root) return;
                    const name = child.name.toLowerCase();
                    if (name.includes("hips") || name.includes("pelvis") || name.includes("root") || name.includes("center") || name.includes("cog")) {
                        root = child;
                    }
                });
                return root;
            };

            const loadGenericModel = async (modelPath, format, customName = null, index = 0, total = 1) => {
                const invalidPaths = ["none", "Export failed", "Error", "undefined", "null", ""];
                if (!modelPath || invalidPaths.includes(modelPath) || modelPath.includes("Error:") || modelPath.includes("failed")) {
                    console.log("[HY-Motion] Skipping invalid 3D model path:", modelPath);
                    return;
                }

                // Ensure engine is initialized (checks both library and scene)
                if (!THREE && window.__HY_MOTION_THREE__) THREE = window.__HY_MOTION_THREE__;

                // ALWAYS ensure initialization runs, because it sets up the SCENE/CAMERA, not just the library
                await ensureInitialized();
                if (!THREE) THREE = window.__HY_MOTION_THREE__;

                console.log("[HY-Motion] Loading 3D Model Path:", modelPath, `(${index + 1}/${total})`);

                // Optimization: Check if this model is already loaded (and it's the same execution)
                if (total === 1 && loadedModels.length === 1 &&
                    loadedModels[0].modelPath === modelPath &&
                    loadedModels[0].timestamp === lastTimestamp) {
                    console.log("[HY-Motion] Model already loaded and up-to-date, skipping redundant reload.");
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
                        const module = await import(FBX_LOADER_URL);
                        LoaderClass = module.FBXLoader;
                        if (!window.fflate) window.fflate = await import("https://esm.sh/fflate@0.8.0");
                    } else if (format === 'glb' || format === 'gltf') {
                        const module = await import(GLTF_LOADER_URL);
                        LoaderClass = module.GLTFLoader;
                    } else if (format === 'obj') {
                        const module = await import(OBJ_LOADER_URL);
                        LoaderClass = module.OBJLoader;
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

                    // Force refresh via timestamp if provided
                    if (lastTimestamp) {
                        fetchUrl += `&t=${lastTimestamp}`;
                    }

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
                            if ((format === 'glb' || format === 'gltf') && result.animations) {
                                fbx.animations = result.animations;
                            }
                            fbx.updateMatrixWorld(true);

                            const debugMat = new T.MeshStandardMaterial({ color: 0xffaa00, roughness: 0.5, metalness: 0.5, side: T.DoubleSide });
                            fbx.traverse((child) => {
                                if (child.isMesh) {
                                    if (!child.material || child.material.type === "MeshBasicMaterial") child.material = debugMat;
                                    child.castShadow = true;
                                    child.receiveShadow = true;
                                    child.frustumCulled = true; // Use default frustum culling for performance
                                }
                            });

                            if (scene) {
                                scene.add(fbx);
                            } else {
                                if (!isInitialized) {
                                    console.log(`[HY-Motion] Node cleaned up while loading ${filename}, ignoring.`);
                                } else {
                                    console.warn("[HY-Motion] Scene undefined but initialized in loadGenericModel callback, skipping add.");
                                }
                            }
                            currentModel = fbx;
                            fbx.userData.modelPath = modelPath; // Store for search

                            // Log animation details
                            if (fbx.animations && fbx.animations.length) {
                                console.log(`[HY-Motion] Loaded ${fbx.animations.length} animations. Duration: ${fbx.animations[0].duration.toFixed(3)}s`);
                            } else {
                                console.log(`[HY-Motion] No animations found in model.`);
                            }

                            loadedModels.push({
                                model: fbx,
                                modelPath: modelPath,
                                timestamp: lastTimestamp,
                                name: customName || filename,
                                fileName: filename,
                                subfolder: subfolder,
                                fileType: type,
                                basePosition: { x: 0, y: 0, z: 0 },
                                baseRotation: { x: 0, y: 0, z: 0 },
                                baseScale: { x: 1, y: 1, z: 1 },
                                isInPlace: false
                            });

                            // CRITICAL: Create hit proxy immediately so mesh is selectable
                            updateHitProxies();

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

                                // Update maxFrames if it's the first model in a batch or if the new animation is longer
                                const animFrames = Math.floor(anims[0].duration * targetFPS);
                                if (maxFrames === 0 || index === 0 || animFrames > maxFrames) {
                                    maxFrames = Math.max(animFrames, 1);
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
                            startAnimating("model-loaded"); // Keep loop active after loading

                            // Auto-centering disabled as per user request
                            /*
                            if (total > 1) {
                                // Center on all models
                                const group = new THREE.Group();
                                loadedModels.forEach(m => group.add(m.model.clone())); // Dummy group for centering auto-box
                                centerCameraOnObject(group);
                            } else {
                                centerCameraOnObject(fbx);
                            }
                            */

                            // Show Transform and Apply buttons for loader
                            if (nodeData.name === "HYMotion3DModelLoader") {
                                applyBtn.style.display = "block";
                                transformBtn.style.display = "block";
                            }

                            // Apply current transform widget values immediately
                            throttledRefreshTransforms();

                            // Force immediate render to prevent black screen
                            if (renderer && scene && camera) {
                                renderer.render(scene, camera);
                            }
                            // Start animation loop for smooth interactions
                            startAnimating();
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

            const ensureInt = (val) => {
                if (Array.isArray(val)) {
                    return parseInt(val[val.length - 1]) || 0;
                }
                return parseInt(val) || 0;
            };

            const ensureString = (val) => {
                if (typeof val === 'string') return val;
                if (Array.isArray(val)) {
                    // Check if it's a character array (e.g. ['f', 'b', 'x'])
                    if (val.every(i => typeof i === 'string' && i.length <= 1)) return val.join('');
                    // Otherwise take the last element (ComfyUI pattern)
                    return String(val[val.length - 1]);
                }
                return String(val);
            };

            const ensureFloat = (val) => {
                if (typeof val === 'number') return val;
                if (Array.isArray(val)) {
                    return parseFloat(val[val.length - 1]) || 0;
                }
                return parseFloat(val) || 0;
            };

            const handleData = async (data) => {
                if (!data) return;

                // Sync timestamp for forced reloads
                const newTimestamp = ensureFloat(data.timestamp);
                const isNewExecution = newTimestamp && newTimestamp !== lastTimestamp;
                if (isNewExecution) {
                    console.log("[HY-Motion] New execution detected, timestamp:", newTimestamp);
                    lastTimestamp = newTimestamp;

                    // Trigger global asset list refresh for ALL viewer nodes
                    refreshAllHyMotionAssets();
                }

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
                    if (urlStr !== lastFbxUrl || isNewExecution) {
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

                // Support for 'fbx' key containing list of dicts (standard ComfyUI format)
                if (data.fbx && Array.isArray(data.fbx)) {
                    data.fbx.forEach((info, i) => {
                        if (info.filename) {
                            const type = info.type || "output";
                            const subfolder = info.subfolder || "";
                            const modelPath = (type === "output" ? "output/" : "input/") + (subfolder ? subfolder + "/" : "") + info.filename;

                            // Prevent reload if already handling this via rawUrl (to avoid double load)
                            if (lastFbxUrl.includes(info.filename)) return;

                            console.log("[HY-Motion] Loading FBX from info dict:", modelPath);
                            loadGenericModel(modelPath, 'fbx', info.filename, i, data.fbx.length);
                        }
                    });
                }

                if (data.model_url) {
                    const url = ensureString(data.model_url);
                    if (url !== lastModelUrl || isNewExecution) {
                        lastModelUrl = url;
                        const format = ensureString(data.format || 'fbx');
                        startFrame = ensureInt(data.start_frame);
                        endFrame = ensureInt(data.end_frame);
                        await loadGenericModel(url, format);
                    }
                }

                // Also refresh visuals if transform changed via Run
                throttledRefreshTransforms();
            };




            // Lazy initialization - only init when data is actually loaded
            // This prevents the 4-second freeze on ComfyUI reload
            const ensureInitialized = async () => {
                if (isInitialized) return;

                // If library is loading, wait for it
                if (isGlobalInitializing) await globalInitPromise;

                // Then perform local initialization if not already done
                if (!isInitialized) {
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
                                throttledRefreshTransforms();
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
            // Consolidated in-place button: global when no selection, per-selection when selected
            inPlaceBtn.onclick = (e) => {
                e.stopPropagation();

                if (selectedModels.length > 0) {
                    // Per-selection mode: toggle in-place for all selected objects
                    const newState = !selectedModels[0].obj.isInPlace;
                    for (const s of selectedModels) {
                        s.obj.isInPlace = newState;
                    }
                    // Update button appearance based on selection state
                    const allInPlace = selectedModels.every(s => s.obj.isInPlace);
                    inPlaceBtn.innerText = allInPlace ? "🧍‍♂️" : "🏃‍♂️";
                    inPlaceBtn.style.background = allInPlace ? "#0066cc" : "#444";
                    inPlaceBtn.title = `In-Place: ${selectedModels.length} selected (${allInPlace ? 'ON' : 'OFF'})`;
                } else {
                    // Global mode: toggle global in-place for all characters
                    isInPlace = !isInPlace;
                    inPlaceBtn.innerText = isInPlace ? "🧍‍♂️" : "🏃‍♂️";
                    inPlaceBtn.style.background = isInPlace ? "#0066cc" : "#444";
                    inPlaceBtn.title = `Global In-Place: ${isInPlace ? 'ON' : 'OFF'}`;
                }
                requestRender();
            };

            // Legacy handler kept for backwards compatibility but does nothing
            selectionInPlaceBtn.onclick = (e) => { e.stopPropagation(); };
            focusBtn.onclick = (e) => {
                e.stopPropagation();
                if (selectedModels.length > 0) {
                    // Focus on the first selected object
                    const s = selectedModels[0];
                    centerCameraOnObject(s.type === 'model' ? s.obj.model : s.obj.group);
                } else if (currentModel) {
                    centerCameraOnObject(currentModel);
                } else if (skeletalSamples.length > 0) {
                    const group = new THREE.Group();
                    skeletalSamples.forEach(s => group.add(s.group.clone()));
                    centerCameraOnObject(group);
                }
            };

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

                    // Apply in-place logic after setting mixer time (same as in animate loop)
                    for (const m of mixers) {
                        const modelObj = loadedModels.find(obj => obj.model === m.getRoot());
                        const activeInPlace = isInPlace || (modelObj && modelObj.isInPlace);

                        if (activeInPlace) {
                            const root = findRootBone(m.getRoot());
                            if (root) {
                                // Lock horizontal local position to 0
                                root.position.x = 0;
                                root.position.z = 0;
                            }
                        }
                    }
                }

                // Sync skeletons for motion data
                updateSkeletons(currentFrame);
                requestRender(); // Trigger single render
            };

            // Proper cleanup when node is removed
            this.onRemoved = function () {
                console.log("[HY-Motion] Removing 3D Viewer Node and cleaning up resources:", node.id);

                // Remove from tracking set
                activeViewerNodes.delete(node.id);
                const remainingViewers = activeViewerNodes.size;
                console.log(`[HY-Motion] Remaining active viewers: ${remainingViewers}`);

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

                // Remove global event listeners (must match the addEventListener flags)
                if (this._handleKeyPress) {
                    document.removeEventListener('keydown', this._handleKeyPress, true);
                    this._handleKeyPress = null;
                }

                // Dispose Three.js resources
                if (scene) {
                    scene.traverse((object) => {
                        disposeObject(object);
                    });
                    scene.clear();
                }
                if (renderer) {
                    if (renderer.domElement && renderer.domElement.parentNode) {
                        renderer.domElement.parentNode.removeChild(renderer.domElement);
                    }
                    renderer.dispose();

                    // Only force context loss if this is the LAST viewer
                    // This prevents breaking other viewers or new nodes
                    if (remainingViewers === 0) {
                        console.log("[HY-Motion] Last viewer removed - forcing WebGL context loss");
                        renderer.forceContextLoss();
                    }
                    renderer = null;
                }

                if (bboxCache) bboxCache.clear();
                scene = null;
                camera = null;
                orbitControls = null;
                if (this.container) this.container = null;
                isInitialized = false;
                isAnimating = false; // Final kill switch for loop
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
            setTimeout(() => {
                const T = window.__HY_MOTION_THREE__;
                if (T && canvasContainer.clientWidth > 0 && canvasContainer.clientHeight > 0) {
                    // Find the canvas element
                    const canvas = canvasContainer.querySelector('canvas');
                    if (canvas) {
                        // Get the renderer from window storage
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
