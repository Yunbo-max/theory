"""Private, source-bound route snapshots on the existing task map; no live feed."""
import html
import _autoresearch as C


COLORS = ["#2563eb", "#b45309", "#047857", "#9333ea", "#be123c", "#0e7490", "#4f46e5", "#6d5b12"]
BACKGROUNDS = ["#eef4ff", "#fef5e7", "#ecf8f1", "#f5effb"]
REGION_LABELS = {"P": "Portfolio", "I": "Intake", "L": "Literature", "B": "Tasks & benchmarks",
    "H": "Ideas & collision", "M": "Method design", "S": "Models & settings", "G": "Gate A",
    "E": "Execution & recovery", "V": "Full validation", "W": "Paper writing", "F": "Figures & acceptance",
    "R": "Submission & rebuttal", "A": "Reproduction & release", "C": "Communication", "X": "Visual navigation"}
# Short display translations; immutable node IDs and source-language titles remain.
NODE_LABELS = {
    "P": ["Goals & portfolio", "Assets & current position", "Priorities & WIP", "Resources & people", "Growth & retrospectives"],
    "I": ["Entry & next decision", "Contribution type", "Venue & stage", "Success & exit"],
    "L": ["Problem & mechanism search", "Topic map & citations", "Primary reading", "Code & reproduction", "Coverage & new evidence"],
    "B": ["Task & metric contract", "Natural failure census", "Strong baseline qualification", "Freeze Parent Problem", "Contribution value gate"],
    "H": ["Failure & hidden assumptions", "Necessity & counterargument", "Atomic claim & prediction", "Functional collision audit", "IPCG & candidate choice"],
    "M": ["Causal chain & intervention", "Math & computability", "Minimum algorithm", "Complexity & boundaries", "Component & proof obligations"],
    "S": ["Benchmark selection", "Data & split audit", "Model & API selection", "Fair comparison settings", "Complete run configuration"],
    "G": ["Decisive experiment", "Frozen rules", "Implementation & baseline", "Prompt / inference / training", "Gate A decision & reroute"],
    "E": ["Pinned environments", "Semantic checks & canaries", "Queue & resources", "Diagnosis & child protocol", "Raw evidence & checkpoints"],
    "V": ["Claim-to-experiment matrix", "Baselines & repeats", "Ablation & replacement", "Generalization & boundaries", "Quality / cost / failures", "Statistics & evidence freeze"],
    "W": ["Template & page budget", "Central claim & narrative", "Main & appendix", "Section blueprint", "Paragraph & citation", "Introduction & related work", "Method / results / discussion", "Title / abstract / conclusion", "Result-driven propagation"],
    "F": ["Method & motivation figure", "Results & editable tables", "Captions & readability", "Compilation & consistency"],
    "R": ["Submission & anonymity", "Review issue ledger", "Rebuttal priorities", "Response & revision", "Decision / retarget / final"],
    "A": ["GitHub reproducibility", "HF models / data / demo", "arXiv version & metadata", "Release manifest", "Feedback & revision"],
    "C": ["Project page & demo", "Poster / talk / video", "X / LinkedIn / Xiaohongshu", "Recipients & email draft", "Homepage / CV / knowledge"],
    "X": ["Position & route exploration", "PDF & figure observation", "Code / logs / web evidence", "Bounded exploration", "Route registration", "Map quality & capability"]}


def escaped(value):
    return html.escape(str(value), quote=True)


def map_svg(view):
    registry = C.load_file(C.SKILL / "assets/research-nodes.json")
    groups, nodes = registry["groups"], {n["id"]: n for n in registry["nodes"]}
    positions, regions = {}, []
    top = 20
    for start in range(0, len(groups), 4):
        row = groups[start:start + 4]
        height = 72 + 30 * max(len(g[3]) for g in row)
        for column, group in enumerate(row):
            x = 16 + column * 306
            regions.append(f'<g data-map-region="{escaped(group[0])}"><title>{escaped(group[1])}</title><rect x="{x}" y="{top}" width="288" height="{height - 14}" rx="14" fill="{BACKGROUNDS[(start // 4 + column) % 4]}" stroke="#d8e1eb"/><text x="{x + 18}" y="{top + 28}" font-size="14" font-weight="650">{escaped(group[0] + " · " + REGION_LABELS.get(group[0], group[0]))}</text></g>')
            for index, item in enumerate(group[3]):
                positions[group[0] + item[0]] = (x + 24, top + 57 + index * 30)
        top += height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1232 {top + 8}" role="img" aria-labelledby="route-map-title" style="width:100%;min-width:760px;font-family:system-ui,sans-serif">',
        '<title id="route-map-title">Research task map: 16 regions, 84 nodes and recorded worker routes</title>',
        '<defs><marker id="map-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>', *regions]

    def curve(first, second):
        x1, y1 = positions[first]; x2, y2 = positions[second]
        bend = 20 if x1 == x2 else (x2 - x1) / 2
        return f'M {x1} {y1} C {x1 + bend} {y1}, {x2 - bend} {y2}, {x2} {y2}'

    for edge in registry["edges"]:
        title = escaped(edge["from"] + " → " + edge["to"] + " · " + edge["when"])
        parts.append(f'<path data-map-edge="{escaped(edge["id"])}" d="{curve(edge["from"], edge["to"])}" fill="none" stroke="#94a3b8" stroke-width="1" opacity=".18" marker-end="url(#map-arrow)"><title>{title}</title></path>')
    for index, route in enumerate(view["routes"]):
        overlay, color = route["overlay"], COLORS[index % len(COLORS)]
        parts.append(f'<g class="route-layer" data-route-index="{index}" fill="none" stroke="{color}" stroke-width="3.5" opacity=".88"><title>{escaped(route["route_id"] + " · " + route["classification"] + " · recorded route")}</title>')
        for step, (first, second) in enumerate(zip(overlay["path"], overlay["path"][1:])):
            parts.append(f'<path data-route-step="{index}-{step}" d="{curve(first, second)}"><title>{escaped(first + " → " + second)}</title></path>')
        if overlay["next_node"]:
            parts.append(f'<path data-planned-step="{index}" d="{curve(overlay["current_node"], overlay["next_node"])}" stroke-dasharray="6 5"><title>Planned next node; not completed</title></path>')
        x, y = positions[overlay["current_node"]]
        parts.append(f'<circle cx="{x}" cy="{y}" r="11" stroke-width="3"/><text x="{x - 17}" y="{y - 13}" stroke="none" fill="{color}" font-size="10">{index + 1}</text></g>')
    for node_id, (x, y) in positions.items():
        node = nodes[node_id]
        labels = NODE_LABELS.get(node_id[0], [])
        index = int(node_id[1:]) - 1
        label = labels[index] if index < len(labels) else node_id
        parts.append(f'<g data-map-node="{escaped(node_id)}"><title>{escaped(node_id + " · " + node["name"])}</title><circle cx="{x}" cy="{y}" r="4" fill="#334155"/><text x="{x + 12}" y="{y + 4}" font-size="11" fill="#243247">{escaped(node_id + " " + label)}</text></g>')
    parts.append('</svg>')
    return "".join(parts)


def render_html(view):
    rows, legend = [], []
    for index, route in enumerate(view["routes"]):
        owner = C.canonical(route["owner"])
        evidence = route["overlay"]["last_evidence"]
        last = evidence["path"] + " / " + evidence["sha256"] if evidence else "unavailable"
        dependency = view["projects"][route["project_id"]]["upstream_validation"]
        budget = C.canonical({"limit": route["budget"], **route["budget_accounting"]})
        values = (route["route_id"], route["project_id"], owner, route["goal"], route["classification"],
            route["checkpoint"]["verification"], route["checkpoint"]["observed_at"],
            " → ".join(route["overlay"]["path"]), route["overlay"]["current_node"],
            route["overlay"]["next_node"] or "unspecified", last, budget,
            C.canonical(dependency) if dependency else ", ".join(route["checkpoint"]["blockers"]) or "none recorded",
            ", ".join(route["reason_codes"]))
        rows.append('<tr>' + ''.join('<td>' + escaped(value) + '</td>' for value in values) + '</tr>')
        legend.append(f'<label><input type="checkbox" checked data-toggle-route="{index}"><span style="color:{COLORS[index % len(COLORS)]}">● {index + 1}. {escaped(route["route_id"])}</span> · {escaped(route["classification"])}</label>')
    headers = ("Route", "Project", "Owner / model", "Goal", "Status", "Verification", "Source time",
               "Recorded path", "Current node", "Planned next", "Last evidence", "Budget", "Upstream / dependencies", "Reasons")
    head = ''.join('<th>' + h + '</th>' for h in headers)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Private research route overview</title>'
        '<style>body{font:15px system-ui,sans-serif;margin:0;background:#f8fafc;color:#1e293b}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:28px;margin-bottom:8px}.muted{color:#52657b;line-height:1.6}.panel{background:white;border:1px solid #dbe3ef;border-radius:16px;margin:20px 0;padding:18px}.map,.table{overflow:auto}label{display:inline-block;margin:5px 18px 5px 0}table{border-collapse:collapse;font-size:12px}td,th{padding:10px;border-bottom:1px solid #dbe3ef;text-align:left;vertical-align:top;max-width:260px;overflow-wrap:anywhere}th{background:#f1f5f9;white-space:nowrap}code{font-size:12px}</style></head><body><main>'
        '<h1>Research route overview</h1><p class="muted">Private snapshot of registered routes. No live chat access. Revision '
        + str(view["revision"]) + ' · observed ' + escaped(view["observed_at"]) + ' · durability: '
        + escaped(view["durability"]["status"]) + '.</p><p class="muted">Coordination blockers: '
        + escaped(', '.join(view["coordination_blockers"]) or 'none recorded') + '</p>'
        '<section class="panel"><h2>Task map and recorded routes</h2><p class="muted">Region backgrounds show task types; grey arrows are conditional map links. Colored paths are recorded worker checkpoints. Rings mark current nodes; dashed segments are planned next steps. Source verification does not certify every reported visit or scientific outcome.</p>'
        + (''.join(legend) or '<p>No registered routes.</p>') + '<div class="map">' + map_svg(view) + '</div></section>'
        '<section class="panel"><h2>Source, evidence and resource details</h2><div class="table"><table><thead><tr>' + head
        + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section></main>'
        '<script>document.querySelectorAll("[data-toggle-route]").forEach(function(input){input.addEventListener("change",function(){document.querySelectorAll("[data-route-index]").forEach(function(layer){if(layer.dataset.routeIndex===input.dataset.toggleRoute){layer.style.display=input.checked?"":"none";}});});});</script></body></html>\n')


def render_mermaid(view):
    lines = ["flowchart TD"]
    for index, route in enumerate(view["routes"]):
        lines.append(f'  subgraph route_group_{index}["{escaped(route["route_id"] + " / " + route["classification"]).replace(chr(10), " ").replace(chr(13), " ")}"]')
        lines.append('    direction TB')
        for step, node in enumerate(route["overlay"]["path"]):
            lines.append(f'    route_{index}_{step}["{escaped(node)}"]')
            if step:
                lines.append(f'    route_{index}_{step - 1} --> route_{index}_{step}')
        if route["overlay"]["next_node"]:
            lines.append(f'    route_{index}_next["{escaped(route["overlay"]["next_node"])} / planned"]')
            lines.append(f'    route_{index}_{len(route["overlay"]["path"]) - 1} -.-> route_{index}_next')
        lines.append('  end')
    return "\n".join(lines) + "\n"
