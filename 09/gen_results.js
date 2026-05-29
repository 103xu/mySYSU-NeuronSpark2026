/**
 * NS-2026-09 完整预测器 (Node.js)
 * 读取 test.jsonl，生成 results.json
 */
const fs = require('fs');
const readline = require('readline');

const GRID_SIZE = 12;
const EVENT_KEYS = ["goal_reached","collision","hazard","box_on_goal","key_collected","portal_used"];
const DIR_VEC = { U: [0, -1], D: [0, 1], L: [-1, 0], R: [1, 0] };
const FIELD_VEC = { '^': [0, -1], 'v': [0, 1], '<': [-1, 0], '>': [1, 0] };

class ProbeLearner {
    static learn(context) {
        const episodes = context.context_episodes;
        const info = {};

        // Terminal stats
        const terminals = episodes.map(e => e.observed_final_terminal);
        const termCounts = {};
        terminals.forEach(t => termCounts[t] = (termCounts[t] || 0) + 1);
        info.common_terminal = Object.entries(termCounts).sort((a,b) => b[1]-a[1])[0][0];

        // Event stats
        info.event_freq = {};
        EVENT_KEYS.forEach(k => {
            let c = 0;
            episodes.forEach(e => { if (e.observed_final_events[k]) c++; });
            info.event_freq[k] = c / episodes.length;
        });

        // Probe event orders and timelines
        info.probe_orders = episodes.map(e => e.observed_final_event_order || []);
        info.probe_timelines = episodes.map(e => e.observed_final_event_timeline || {});

        // Map features
        const full = context.initial_full_grid.join('');
        info.n_ice = (full.match(/I/g) || []).length;
        info.n_portal = (full.match(/P/g) || []).length;
        info.n_field = (full.match(/[\^v<>]/g) || []).length;

        return info;
    }
}

class Simulator {
    constructor(staticGrid, initialDir) {
        this.sgrid = staticGrid;
        this.grid = staticGrid.map(r => [...r]);
        this.agentDir = initialDir || 'D';
        this.stepCount = 0;
        this.events = {};
        this.eventStep = {};
        this.eventOrder = [];
        this.keysHeld = 0;
        this.portalCD = 0;
        this.keysCollected = new Set();
        EVENT_KEYS.forEach(k => { this.events[k] = false; this.eventStep[k] = -1; });
    }

    inBounds(x, y) { return x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE; }

    getStatic(x, y) {
        if (!this.inBounds(x, y)) return '#';
        if (this.keysCollected.has(x+','+y)) return '.';
        return this.sgrid[y][x];
    }

    get(x, y) {
        if (!this.inBounds(x, y)) return '#';
        const d = this.grid[y][x];
        if (d !== '.') return d;
        return this.getStatic(x, y);
    }

    trigger(event) {
        if (!this.events[event]) {
            this.events[event] = true;
            this.eventStep[event] = this.stepCount;
            this.eventOrder.push(event);
        }
    }

    findAgent() {
        for (let y = 0; y < GRID_SIZE; y++)
            for (let x = 0; x < GRID_SIZE; x++)
                if (this.grid[y][x] === 'A') return [x, y];
        return null;
    }

    findPortalDest(x, y) {
        let best = null, bestDist = 999;
        for (let py = 0; py < GRID_SIZE; py++)
            for (let px = 0; px < GRID_SIZE; px++)
                if (this.getStatic(px, py) === 'P' && (px !== x || py !== y)) {
                    const d = Math.abs(px - x) + Math.abs(py - y);
                    if (d < bestDist) { bestDist = d; best = [px, py]; }
                }
        return best;
    }

    processOrbFields() {
        for (let y = 0; y < GRID_SIZE; y++) {
            for (let x = 0; x < GRID_SIZE; x++) {
                if (this.grid[y][x] === 'O') {
                    const sc = this.getStatic(x, y);
                    if ('^v<>'.includes(sc)) {
                        const [dx, dy] = FIELD_VEC[sc];
                        const nx = x + dx, ny = y + dy;
                        if (this.inBounds(nx, ny) && !('#DBO'.includes(this.get(nx, ny))) && this.grid[ny][nx] === '.') {
                            this.grid[y][x] = '.';
                            this.grid[ny][nx] = 'O';
                        }
                    }
                }
            }
        }
    }

    checkPos() {
        const a = this.findAgent();
        if (!a) return;
        const [x, y] = a;
        const sc = this.getStatic(x, y);
        if (sc === 'G') this.trigger('goal_reached');
        if (sc === 'H') this.trigger('hazard');
        if (sc === 'K' && !this.keysCollected.has(x+','+y)) {
            this.trigger('key_collected');
            this.keysHeld++;
            this.keysCollected.add(x+','+y);
        }
    }

    step(action) {
        this.stepCount++;
        this.processOrbFields();

        if (action === 'WAIT') { this.checkPos(); return; }

        const [dx, dy] = DIR_VEC[action] || [0, 0];
        if (dx === 0 && dy === 0) return;

        this.agentDir = action;
        const a = this.findAgent();
        if (!a) return;
        const [ax, ay] = a;
        let tx = ax + dx, ty = ay + dy;

        if (!this.inBounds(tx, ty)) { this.trigger('collision'); return; }

        const target = this.get(tx, ty);
        if (target === '#') { this.trigger('collision'); return; }

        if (target === 'D') {
            if (this.keysHeld > 0) { this.keysHeld--; }
            else { this.trigger('collision'); return; }
        }

        if (target === 'B') {
            const bnx = tx + dx, bny = ty + dy;
            const bc = this.inBounds(bnx, bny) ? this.get(bnx, bny) : '#';
            if (bc === '#' || bc === 'D' || bc === 'B' || (this.inBounds(bnx, bny) && this.grid[bny][bnx] === 'B')) {
                this.trigger('collision'); return;
            }
            this.grid[ty][tx] = '.';
            this.grid[bny][bnx] = 'B';
            if (this.getStatic(bnx, bny) === 'O') this.trigger('box_on_goal');
        }

        // Move agent
        this.grid[ay][ax] = '.';
        this.grid[ty][tx] = 'A';

        // Portal
        if (this.getStatic(tx, ty) === 'P' && this.portalCD <= 0) {
            const dest = this.findPortalDest(tx, ty);
            if (dest && !('#DB'.includes(this.get(dest[0], dest[1])))) {
                this.grid[ty][tx] = '.';
                this.grid[dest[1]][dest[0]] = 'A';
                this.trigger('portal_used');
                this.portalCD = 3;
                tx = dest[0]; ty = dest[1];
            }
        }
        if (this.portalCD > 0) this.portalCD--;

        // Ice slide (1 step only)
        if (this.getStatic(tx, ty) === 'I') {
            const nx = tx + dx, ny = ty + dy;
            if (this.inBounds(nx, ny)) {
                const t2 = this.get(nx, ny);
                if (!('#DB'.includes(t2)) && !(t2 === 'B' && (!this.inBounds(nx+dx, ny+dy) || '#DB'.includes(this.get(nx+dx, ny+dy)) || this.grid[ny+dy] && this.grid[ny+dy][nx+dx] === 'B'))) {
                    if (t2 === 'B') {
                        const bnx = nx + dx, bny = ny + dy;
                        if (this.inBounds(bnx, bny) && !('#DB'.includes(this.get(bnx, bny)))) {
                            this.grid[ny][nx] = '.';
                            this.grid[bny][bnx] = 'B';
                            if (this.getStatic(bnx, bny) === 'O') this.trigger('box_on_goal');
                        } else { this.checkPos(); return; }
                    }
                    this.grid[ty][tx] = '.';
                    this.grid[ny][nx] = 'A';
                    tx = nx; ty = ny;

                    // Portal after ice slide
                    if (this.getStatic(tx, ty) === 'P' && this.portalCD <= 0) {
                        const dest = this.findPortalDest(tx, ty);
                        if (dest && !('#DB'.includes(this.get(dest[0], dest[1])))) {
                            this.grid[ty][tx] = '.';
                            this.grid[dest[1]][dest[0]] = 'A';
                            this.trigger('portal_used');
                            this.portalCD = 3;
                            tx = dest[0]; ty = dest[1];
                        }
                    }
                }
            }
        }

        this.checkPos();
    }

    run(actions, horizon) {
        const limit = horizon > 0 ? horizon : actions.length;
        const steps = Math.min(limit, actions.length);

        for (let i = 0; i < steps; i++) {
            this.step(actions[i]);
            if (this.events.goal_reached || this.events.hazard) break;
        }

        const finalGrid = this.grid.map(r => r.join(''));

        let terminal;
        if (this.events.goal_reached) terminal = 'goal';
        else if (this.events.hazard) terminal = 'hazard';
        else if (this.events.collision) terminal = 'blocked';
        else terminal = 'timeout';

        const effH = Math.max(horizon || steps, steps, 1);
        const timeline = {};
        for (const k of EVENT_KEYS) {
            const s = this.eventStep[k];
            if (s < 0) timeline[k] = 'never';
            else {
                const ratio = s / effH;
                timeline[k] = ratio < 0.33 ? 'early' : ratio < 0.67 ? 'mid' : 'late';
            }
        }

        const order = [...this.eventOrder];
        while (order.length < 3) order.push('none');

        return {
            final_grid: finalGrid,
            events: { ...this.events },
            event_timeline: timeline,
            event_order: order.slice(0, 3),
            terminal: terminal,
            simulator_events: { ...this.events },
            simulator_order: [...this.eventOrder],
        };
    }
}

function predictQuery(initialGrid, query, probeInfo) {
    const initDir = (query.initial_observation && query.initial_observation.sensor && query.initial_observation.sensor.agent_dir) || 'D';
    const sim = new Simulator(initialGrid, initDir);
    const result = sim.run(query.future_actions, query.query_horizon);

    // Post-processing with probe info
    let events = { ...result.events };
    let terminal = result.terminal;

    // Collision correction from probes
    if (probeInfo.event_freq.collision >= 0.8 && !events.collision) {
        events.collision = true;
    }

    // If probe never had an event, be skeptical
    for (const k of EVENT_KEYS) {
        if (probeInfo.event_freq[k] === 0 && events[k]) {
            events[k] = false;
        }
    }

    // Timeline with probe fallback
    const timeline = {};
    for (const k of EVENT_KEYS) {
        if (events[k]) {
            let t = result.event_timeline[k];
            if (t === 'never') {
                const probeTls = probeInfo.probe_timelines
                    .map(pt => pt[k]).filter(tl => tl && tl !== 'never');
                if (probeTls.length > 0) {
                    const counts = {};
                    probeTls.forEach(tl => counts[tl] = (counts[tl] || 0) + 1);
                    t = Object.entries(counts).sort((a,b) => b[1]-a[1])[0][0];
                } else {
                    t = 'early';
                }
            }
            timeline[k] = t;
        } else {
            timeline[k] = 'never';
        }
    }

    // Event order
    let order = result.simulator_order.filter(e => events[e]);
    if (order.length < 3) {
        for (const po of probeInfo.probe_orders) {
            for (const e of po) {
                if (e !== 'none' && events[e] && !order.includes(e) && order.length < 3) {
                    order.push(e);
                }
            }
        }
    }
    while (order.length < 3) order.push('none');

    // Terminal correction
    if (terminal === 'timeout') {
        const ef = probeInfo.event_freq;
        if (ef.collision >= 0.8 || ef.goal_reached > 0 || ef.hazard > 0) {
            if (ef.goal_reached > 0) terminal = 'goal';
            else if (ef.hazard > 0) terminal = 'hazard';
            else terminal = probeInfo.common_terminal;
        }
    }

    return {
        id: query.query_id,
        final_grid: result.final_grid,
        events: events,
        event_timeline: timeline,
        event_order: order.slice(0, 3),
        terminal: terminal,
    };
}

async function main() {
    console.log('NS-2026-09 预测器启动...');
    console.log('读取 test.jsonl...');

    const results = [];
    let ctxCount = 0;
    let queryCount = 0;

    const stream = fs.createReadStream('test.jsonl');
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

    for await (const line of rl) {
        if (!line.trim()) continue;
        const context = JSON.parse(line);
        ctxCount++;

        const probeInfo = ProbeLearner.learn(context);
        const initialGrid = context.initial_full_grid;

        for (const query of context.queries) {
            const prediction = predictQuery(initialGrid, query, probeInfo);
            results.push(prediction);
            queryCount++;
        }

        if (ctxCount % 50 === 0) {
            console.log(`  已处理 ${ctxCount} contexts, ${queryCount} queries...`);
        }
    }

    console.log(`总计: ${ctxCount} contexts, ${queryCount} queries`);

    // Write results
    const outPath = 'results.json';
    fs.writeFileSync(outPath, JSON.stringify(results, null, 2), 'utf-8');
    console.log(`已写入 ${results.length} 条预测到 ${outPath}`);

    // Quick format check
    console.log('\n快速格式检查...');
    let errors = 0;
    const requiredEvents = EVENT_KEYS;
    const validTimeline = ['never', 'early', 'mid', 'late'];
    const validTerminal = ['goal', 'hazard', 'blocked', 'active', 'timeout'];

    for (const r of results) {
        if (!r.id || typeof r.id !== 'string') { errors++; continue; }
        if (!Array.isArray(r.final_grid) || r.final_grid.length !== GRID_SIZE) { errors++; continue; }
        for (const row of r.final_grid) {
            if (typeof row !== 'string' || row.length !== GRID_SIZE) { errors++; break; }
        }
        if (!r.events || typeof r.events !== 'object') { errors++; continue; }
        for (const k of requiredEvents) {
            if (typeof r.events[k] !== 'boolean') { errors++; }
        }
        if (!r.event_timeline || typeof r.event_timeline !== 'object') { errors++; continue; }
        for (const k of requiredEvents) {
            if (!validTimeline.includes(r.event_timeline[k])) { errors++; }
        }
        if (!Array.isArray(r.event_order) || r.event_order.length !== 3) { errors++; continue; }
        if (!validTerminal.includes(r.terminal)) { errors++; }
    }

    if (errors === 0) {
        console.log('格式检查通过!');
    } else {
        console.log(`格式问题: ${errors} 个`);
    }
}

main().catch(err => { console.error('Error:', err); process.exit(1); });
