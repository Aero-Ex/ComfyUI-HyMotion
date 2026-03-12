import { app } from "../../../scripts/app.js";

console.log("[JoeDownloadFile] Extension script loaded");

app.registerExtension({
    name: "Joe.DownloadFile",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "JoeDownloadFile") return;

        console.log("[JoeDownloadFile] beforeRegisterNodeDef for", nodeData.name);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            console.log("[JoeDownloadFile] Node created, id:", node.id);

            // 调整节点高度以容纳 iframe（宽度保持默认，高度约 180）
            try {
                //const defaultWidth = Array.isArray(node.size) ? node.size[0] : 180;
                node.size = [400, 250];
            } catch (e) {
                console.warn("[JoeDownloadFile] Failed to resize node:", e);
            }

            // 创建 iframe，加载 JoeDownloadFile.html
            const iframe = document.createElement("iframe");
            iframe.src = "/extensions/ComfyUI-HyMotion/JoeDownloadFile.html";
            iframe.style.cssText = "width:100%; height:120px; border:0; background:#111;";

            // 尝试使用 ComfyUI 扩展 API 挂载 DOM Widget
            if (typeof node.addDOMWidget === "function") {
                console.log("[JoeDownloadFile] Using addDOMWidget to attach iframe");
                node.addDOMWidget("Download", "joe_download", iframe);
            } else {
                // 兼容老版本：直接挂到 node 的 html 属性
                console.log("[JoeDownloadFile] addDOMWidget not available, using node.html");
                node.html = iframe;
            }

            // 包一层，方便安全调用 postMessage
            function updateIframe(output) {
                if (!output) {
                    console.log("[JoeDownloadFile] updateIframe called with empty output");
                    return;
                }

                // ComfyUI 对带 ui 的节点，会把 ui 字段「摊平」传给 onExecuted，
                // 所以这里既兼容 output.ui 也兼容直接就是 ui 对象的情况。
                const ui = output.ui || output;

                if (!ui) {
                    console.log("[JoeDownloadFile] updateIframe: no usable ui data on output", output);
                    return;
                }
                if (!iframe.contentWindow) {
                    console.log("[JoeDownloadFile] updateIframe: iframe.contentWindow not ready yet");
                    return;
                }

                const fileName = (ui.file_name && ui.file_name[0]) || null;
                const downloadId = (ui.download_id && ui.download_id[0]) || null;
                const fileExists = ui.file_exists ? !!ui.file_exists[0] : false;
                const fileSizeBytes = ui.file_size_bytes ? ui.file_size_bytes[0] : null;
                const filePath = ui.file_path ? ui.file_path[0] : null;

                // console.log("[JoeDownloadFile] Sending to iframe:", {
                //     fileName,
                //     downloadId,
                //     fileExists,
                //     fileSizeBytes,
                // });

                try {
                    iframe.contentWindow.postMessage({
                        type: "loadFileForDownload",
                        fileName,
                        downloadId,
                        fileExists,
                        fileSizeBytes,
                        filePath,
                    }, "*");
                } catch (e) {
                    console.warn("[JoeDownloadFile] Failed to send data to iframe:", e);
                }
            }

            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (output) {
                console.log("[JoeDownloadFile] onExecuted output:", output);
                if (origOnExecuted) {
                    try {
                        origOnExecuted.apply(this, arguments);
                    } catch (e) {
                        console.warn("[JoeDownloadFile] original onExecuted error:", e);
                    }
                }
                updateIframe(output);
            };

            return r;
        };
    },
});


