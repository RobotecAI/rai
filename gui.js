const net = require("net")
const http = require("http")

const tcpPort = 9000
const httpPort = 9001

const nodes = {}
const edges = {}
const logs = []
// Event buffer for frontend
const events = []
// Track active calls to match open/close (FIFO per link)
const activeCalls = {}
let eventIdCounter = 0
let reqIdCounter = 0

// Regex for the new log format: <NAMESPACE:NODE> ACTION TOPIC STATUS
const LOG_RE = /^<([^:]+):([^>]+)> ([^ ]+) (.+) (opened|closed)$/
// Regex for legacy format: NODE: ACTION TOPIC
const OLD_RE = /^([^:]+): ([^ ]+) (.+)$/

function ensureNode(name, kind, namespace = "default") {
    if (!nodes[name]) {
        nodes[name] = { id: name, kind, namespace, pulse: 0 }
    } else {
        // Update namespace if we discover it (and it was previously default)
        if (nodes[name].namespace === "default" && namespace !== "default") {
            nodes[name].namespace = namespace
        }
    }
    nodes[name].pulse++
}

function ensureEdge(source, target) {
    const key = source + ">" + target
    if (!edges[key]) {
        edges[key] = { source, target, pulse: 0 }
    }
    edges[key].pulse++
}

function addEvent(type, source, target, action, reqId) {
    const evt = {
        id: ++eventIdCounter,
        type,
        source,
        target,
        action,
        reqId,
        time: Date.now()
    }
    events.push(evt)
    if (events.length > 500) events.shift()
}

const tcpServer = net.createServer(socket => {
    socket.on("data", data => {
        data
            .toString()
            .split("\n")
            .filter(Boolean)
            .forEach(line => {
                let namespace = "default"
                let nodeName = ""
                let action = ""
                let topic = ""
                let status = "" // opened or closed or undefined (legacy)

                let match = line.match(LOG_RE)
                if (match) {
                    namespace = match[1]
                    nodeName = match[2]
                    action = match[3]
                    topic = match[4]
                    status = match[5]
                } else if ((match = line.match(OLD_RE))) {
                    nodeName = match[1]
                    action = match[2]
                    topic = match[3]
                    // Legacy doesn't have status, treat as one-shot (opened)
                    // We can simulate an immediate close or just one event
                } else {
                    // Fallback
                    const sep = line.indexOf(":")
                    if (sep !== -1) {
                        nodeName = line.slice(0, sep).trim()
                        const payload = line.slice(sep + 1).trim()
                        const sendMatch = payload.match(/^send_message ([^ ]+)$/)
                        const recvMatch = payload.match(/^receive_message ([^ ]+)$/)
                        // Match register command: <NS:NODE> NODE register
                        // But wait, the standard parser might miss it if it doesn't match LOG_RE
                        // LOG_RE expects 5 groups: <NS:NODE> ACTION TOPIC STATUS
                        // The register log is: <NS:NODE> NODE register
                        // It has 3 parts after the bracket if we consider "register" as action?
                        // Actually: <ns:name> name register
                        // Let's try to match it specifically

                        const regMatch = line.match(/^<([^:]+):([^>]+)> ([^ ]+) register$/)
                        if (regMatch) {
                            namespace = regMatch[1]
                            nodeName = regMatch[2] // This is the node name from prefix
                            // The 3rd group is also node name usually
                            ensureNode(nodeName, "connector", namespace)
                            return // Done
                        }

                        if (sendMatch) {
                            action = "send_message"
                            topic = sendMatch[1]
                        } else if (recvMatch) {
                            action = "receive_message"
                            topic = recvMatch[1]
                        }

                        // Handle generic action if parsed from log
                        if (!action && match && match[3]) {
                             action = match[3]
                        }
                    }
                }

                if (!action || !topic || !nodeName) return

                // Ensure nodes exist.
                // The nodeName (log source) definitely belongs to the parsed namespace.
                ensureNode(nodeName, "connector", namespace)

                // If action is create_service, the topic (service name) belongs to this namespace
                if (action === "create_service") {
                    ensureNode(topic, "topic", namespace)
                } else {
                    // The topic/target does NOT necessarily belong to the same namespace.
                    // Keep it default to avoid incorrect coloring (e.g. all topics becoming rai_megamind).
                    ensureNode(topic, "topic", "default")
                }

                let source = nodeName
                let target = topic

                if (action === "receive_message") {
                    source = topic
                    target = nodeName
                }

                ensureEdge(source, target)

                // Handle events
                const linkKey = source + ">" + target

                if (status === "opened") {
                    const reqId = ++reqIdCounter
                    if (!activeCalls[linkKey]) activeCalls[linkKey] = []
                    activeCalls[linkKey].push(reqId)
                    addEvent("start", source, target, action, reqId)
                } else if (status === "closed") {
                    // Find matching request
                    if (activeCalls[linkKey] && activeCalls[linkKey].length > 0) {
                        const reqId = activeCalls[linkKey].shift()
                        addEvent("end", source, target, action, reqId)
                    } else {
                        // Orphan closed? Just ignore or log
                    }
                } else {
                    // Legacy or no status - treat as atomic pulse
                    // Generate a start and immediate end?
                    // Or just use the old pulse mechanism.
                    // But we want to support the user request.
                    // Let's generate a self-contained event sequence for legacy
                    const reqId = ++reqIdCounter
                    addEvent("start", source, target, action, reqId)
                    setTimeout(() => {
                         addEvent("end", source, target, action, reqId)
                    }, 500) // Fake duration for legacy
                }

                logs.push({
                    time: new Date().toLocaleTimeString(),
                    line
                })

                if (logs.length > 300) logs.shift()
                console.log(line)
            })
    })
})

tcpServer.listen(tcpPort, () => {
    console.log("TCP log server listening on port", tcpPort)
})

const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>Observability Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            background: #0b0b0b;
            color: #ddd;
            font-family: monospace;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 100vh;
            box-sizing: border-box;
        }

        #graph-container {
            flex: 1;
            border: 1px solid #333;
            position: relative;
            overflow: hidden;
            background: #111;
        }

        svg {
            width: 100%;
            height: 100%;
        }

        #logs {
            height: 150px;
            overflow-y: auto;
            border: 1px solid #333;
            margin-top: 10px;
            padding: 10px;
            background: #111;
            font-size: 12px;
        }

        .node circle {
            stroke: #fff;
            stroke-width: 1.5px;
        }

        .node text {
            fill: #aaa;
            font-size: 12px;
            pointer-events: none;
            text-shadow: 0 1px 0 #000, 1px 0 0 #000, 0 -1px 0 #000, -1px 0 0 #000;
        }

        .link {
            stroke: #666;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
        }

        .link.pulse {
            stroke: #4cff4c;
            stroke-width: 3px;
        }

        .node.pulse circle {
            stroke: #4cff4c;
            stroke-width: 3px;
        }

        .group-hull {
            fill-opacity: 0.1;
            stroke-width: 20;
            stroke-linejoin: round;
            stroke-opacity: 0.2;
        }

        .group-label {
            fill: #666;
            font-size: 24px;
            font-weight: bold;
            opacity: 0.4;
            pointer-events: none;
            text-anchor: middle;
            dominant-baseline: middle;
        }
    </style>
</head>
<body>

<div style="position: absolute; top: 20px; right: 20px; z-index: 1000;">
    <button onclick="resetGraph()" style="background: #333; color: #ddd; border: 1px solid #555; padding: 8px 12px; cursor: pointer; font-family: monospace;">Reset</button>
</div>

<h2>Data Flow Graph (D3.js)</h2>
<div id="graph-container"></div>
<h2>Recent Events</h2>
<div id="logs"></div>

<script>
    const width = document.getElementById("graph-container").clientWidth;
    const height = document.getElementById("graph-container").clientHeight;

    const svg = d3.select("#graph-container").append("svg")
        .attr("viewBox", [0, 0, width, height]);

    // Arrow marker
    svg.append("defs").append("marker")
        .attr("id", "arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 25) // Offset to not overlap with node
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("fill", "#666")
        .attr("d", "M0,-5L10,0L0,5");

    const g = svg.append("g");

    // Layers
    const hullGroup = g.append("g").attr("class", "hulls");
    const labelGroup = g.append("g").attr("class", "labels");
    const linkGroup = g.append("g").attr("class", "links");
    const particleGroup = g.append("g").attr("class", "particles");
    const nodeGroup = g.append("g").attr("class", "nodes");

    // Zoom behavior
    const zoom = d3.zoom()
        .filter(event => !event.shiftKey)
        .extent([[0, 0], [width, height]])
        .scaleExtent([0.1, 8])
        .on("zoom", ({ transform }) => {
            g.attr("transform", transform)
        })

    svg.call(zoom)

    let simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(30).iterations(2));

    let nodesData = [];
    let linksData = [];
    let processedEventIds = new Set();
    const activeParticles = {}; // Map reqId -> selection
    let lastNodePulse = {};

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    function update() {
        fetch("/state")
            .then(res => res.json())
            .then(data => {
                const newNodes = data.nodes;
                const newLinks = data.edges;

                // Update logs
                const logsEl = document.getElementById("logs");
                logsEl.innerHTML = "";
                data.logs.slice().reverse().forEach(l => {
                    const d = document.createElement("div");
                    d.textContent = "[" + l.time + "] " + l.line;
                    logsEl.appendChild(d);
                });

                // Check for structural changes
                let structureChanged = false;
                if (newNodes.length !== nodesData.length || newLinks.length !== linksData.length) {
                    structureChanged = true;
                }

                // Merge data (keep existing simulation nodes to preserve position)
                const nodeMap = new Map(nodesData.map(d => [d.id, d]));
                nodesData = newNodes.map(n => {
                    const existing = nodeMap.get(n.id);
                    if (existing) {
                        existing.kind = n.kind;
                        existing.namespace = n.namespace;
                        existing.pulse = n.pulse;
                        // Keep fixed position if set
                        if (existing.fx != null) n.fx = existing.fx;
                        if (existing.fy != null) n.fy = existing.fy;
                        return existing;
                    }
                    structureChanged = true;
                    return { ...n, x: width/2, y: height/2 };
                });

                const linkMap = new Map(linksData.map(d => [d.source.id + ">" + d.target.id, d]));
                linksData = newLinks.map(l => {
                    const key = l.source + ">" + l.target;
                    const existing = linkMap.get(key);
                    if (existing) {
                        return existing;
                    }
                    structureChanged = true;
                    return { ...l };
                });

                render(structureChanged);

                // Handle events
                if (data.events) {
                    data.events.forEach(evt => {
                        if (processedEventIds.has(evt.id)) return;
                        processedEventIds.add(evt.id);
                        handleEvent(evt);
                    });

                    // Cleanup old IDs to prevent memory leak
                    if (processedEventIds.size > 2000) {
                        processedEventIds = new Set(Array.from(processedEventIds).slice(-1000));
                    }
                }
            });
    }

    function handleEvent(evt) {
        // Find source and target nodes (they should exist by now)
        const sourceNode = nodesData.find(n => n.id === evt.source);
        const targetNode = nodesData.find(n => n.id === evt.target);
        if (!sourceNode || !targetNode) return;

        const isService = (evt.action === "service_call" || evt.action === "call_service");
        const color = isService ? "#ff4c4c" : "#4cff4c"; // Red for service, Green for msg

        if (evt.type === "start") {
            // Spawn particle
            const particle = particleGroup.append("circle")
                .attr("r", 4)
                .attr("fill", color)
                .attr("cx", sourceNode.x)
                .attr("cy", sourceNode.y);

            // Store reference
            activeParticles[evt.reqId] = { el: particle, target: targetNode };

            // Animate to 80% and hold
            // We use a custom tween or just transition to a point near target
            const midX = sourceNode.x + (targetNode.x - sourceNode.x) * 0.9;
            const midY = sourceNode.y + (targetNode.y - sourceNode.y) * 0.9;

            particle.transition()
                .duration(1000)
                .ease(d3.easeCubicOut)
                .attr("cx", midX)
                .attr("cy", midY);

        } else if (evt.type === "end") {
            const p = activeParticles[evt.reqId];
            if (p) {
                // Complete animation to target
                p.el.interrupt(); // Stop holding/moving
                p.el.transition()
                    .duration(200)
                    .ease(d3.easeLinear)
                    .attr("cx", targetNode.x)
                    .attr("cy", targetNode.y)
                    .remove();

                delete activeParticles[evt.reqId];
            }
        }
    }

    function render(structureChanged) {
        // Update simulation
        simulation.nodes(nodesData);
        simulation.force("link").links(linksData);

        // Grouping force
        simulation.force("group", alpha => {
            const centroids = {};
            nodesData.forEach(d => {
                if (!centroids[d.namespace]) centroids[d.namespace] = {x: 0, y: 0, count: 0};
                centroids[d.namespace].x += d.x;
                centroids[d.namespace].y += d.y;
                centroids[d.namespace].count++;
            });
            for (let k in centroids) {
                centroids[k].x /= centroids[k].count;
                centroids[k].y /= centroids[k].count;
            }
            nodesData.forEach(d => {
                const c = centroids[d.namespace];
                if (!c) return;
                d.vx += (c.x - d.x) * alpha * 0.1;
                d.vy += (c.y - d.y) * alpha * 0.1;
            });
        });

        // Only restart simulation if structure changed
        if (structureChanged) {
            simulation.alpha(0.3).restart();
        }

        // Draw Links
        const link = linkGroup.selectAll(".link")
            .data(linksData, d => d.source.id + ">" + d.target.id)
            .join("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrow)");

        // Pulse nodes
        const node = nodeGroup.selectAll(".node")
            .data(nodesData, d => d.id)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .filter(event => event.shiftKey) // Only drag node if Shift is pressed
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.selectAll("circle").remove();
        node.append("circle")
            .attr("r", 20)
            .attr("fill", d => d.kind === "topic" ? "#222" : colorScale(d.namespace))
            .attr("stroke", d => d.kind === "topic" ? "#4ca3ff" : "#4cff4c");

        node.selectAll("text").remove();
        node.append("text")
            .attr("dy", 32)
            .attr("text-anchor", "middle")
            .text(d => d.id);

         node.each(function(d) {
              if ((lastNodePulse[d.id] || 0) < d.pulse) {
                  d3.select(this).select("circle")
                    .transition().duration(250).attr("fill", "#fff")
                    .transition().duration(750).attr("fill", d.kind === "topic" ? "#222" : colorScale(d.namespace));
                  lastNodePulse[d.id] = d.pulse;
              }
         });

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("transform", d => \`translate(\${d.x},\${d.y})\`);

            // Draw hulls
            const pointsByNs = d3.group(nodesData, d => d.namespace);
            const hulls = [];
            for (const [ns, nodes] of pointsByNs) {
                if (ns === "default" && nodes.length < 2) continue;
                if (nodes.length < 3) continue;
                const points = nodes.map(d => [d.x, d.y]);
                const hull = d3.polygonHull(points);
                if (hull) hulls.push({ns, path: hull});
            }

            hullGroup.selectAll("path")
                .data(hulls)
                .join("path")
                .attr("class", "group-hull")
                .attr("fill", d => colorScale(d.ns))
                .attr("stroke", d => colorScale(d.ns))
                .attr("d", d => "M" + d.path.join("L") + "Z");

            // Draw namespace labels
            const labels = [];
            for (const [ns, nodes] of pointsByNs) {
                // if (ns === "default" && nodes.length < 2) continue;
                // if (nodes.length < 3) continue;

                let x = 0, y = 0;
                nodes.forEach(n => { x += n.x; y += n.y; });
                x /= nodes.length;
                y /= nodes.length;
                labels.push({ns, x, y});
            }

            labelGroup.selectAll("text")
                .data(labels)
                .join("text")
                .attr("class", "group-label")
                .attr("x", d => d.x)
                .attr("y", d => d.y)
                .text(d => d.ns);

            // Update particle positions if we need to track moving nodes
            // The transitions handle x/y, but if nodes move during transition, particles might detach
            // Simple approach: transitions target fixed coords. If nodes move, particles might miss.
            // But since nodes are stationary mostly, it's acceptable.
        });
    }

    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
    }

    function resetGraph() {
        fetch("/reset").then(() => {
            nodesData = [];
            linksData = [];
            processedEventIds = new Set();
            for (let k in activeParticles) {
                activeParticles[k].el.remove();
                delete activeParticles[k];
            }
            lastNodePulse = {};
            lastEdgePulse = {};

            // Clear SVGs
            nodeGroup.selectAll("*").remove();
            linkGroup.selectAll("*").remove();
            hullGroup.selectAll("*").remove();
            labelGroup.selectAll("*").remove();
            particleGroup.selectAll("*").remove();

            update();
        });
    }

    setInterval(update, 500);
    update();
</script>

</body>
</html>
`

const httpServer = http.createServer((req, res) => {
    if (req.url === "/reset") {
        for (const key in nodes) delete nodes[key]
        for (const key in edges) delete edges[key]
        logs.length = 0
        events.length = 0
        for (const key in activeCalls) delete activeCalls[key]
        res.writeHead(200, { "Content-Type": "text/plain" })
        res.end("Reset")
        return
    }

    if (req.url === "/state") {
        res.writeHead(200, { "Content-Type": "application/json" })
        res.end(JSON.stringify({
            nodes: Object.values(nodes),
            edges: Object.values(edges),
            logs,
            events // Send recent events
        }))
        return
    }

    res.writeHead(200, { "Content-Type": "text/html" })
    res.end(htmlContent)
})

httpServer.listen(httpPort, () => {
    console.log("Web viewer available on port", httpPort)
})
