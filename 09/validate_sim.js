/**
 * Node.js 物理模拟验证
 * 在训练数据上测试模拟器的准确性
 */
const fs = require('fs');
const readline = require('readline');

const GRID_SIZE = 12;
const DIR_VEC = { U: [0, -1], D: [0, 1], L: [-1, 0], R: [1, 0] };
const FIELD_VEC = { '^': [0, -1], 'v': [0, 1], '<': [-1, 0], '>': [1, 0] };
const EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"];

function cloneGrid(grid) {
    return grid.map(row => [...row]);
}

class Simulator {
    constructor(staticGrid, initialDir = 'D') {
        this.sgrid = staticGrid;
        this.grid = staticGrid.map(r => [...r]);
        this.agentDir = initialDir;
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
        const key = `${x},${y}`;
        if (this.keysCollected.has(key)) return '.';
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

    checkPos() {
        const a = this.findAgent();
        if (!a) return;
        const [x, y] = a;
        const sc = this.getStatic(x, y);
        if (sc === 'G') this.trigger('goal_reached');
        if (sc === 'H') this.trigger('hazard');
        if (sc === 'K' && !this.keysCollected.has(`${x},${y}`)) {
            this.trigger('key_collected');
            this.keysHeld++;
            this.keysCollected.add(`${x},${y}`);
        }
    }

    step(action) {
        this.stepCount++;

        // 全局场效应：移动所有在场上的实体
        this.processGlobalFields();

        if (action === 'WAIT') {
            this.checkPos();
            return;
        }

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
            if (dest && this.get(dest[0], dest[1]) !== '#' && this.get(dest[0], dest[1]) !== 'D' && this.get(dest[0], dest[1]) !== 'B') {
                this.grid[ty][tx] = '.';
                this.grid[dest[1]][dest[0]] = 'A';
                this.trigger('portal_used');
                this.portalCD = 3;
                tx = dest[0]; ty = dest[1];
            }
        }
        if (this.portalCD > 0) this.portalCD--;

        // Ice slide
        while (this.getStatic(tx, ty) === 'I') {
            const nx = tx + dx, ny = ty + dy;
            if (!this.inBounds(nx, ny)) break;
            const t2 = this.get(nx, ny);
            if (t2 === '#' || t2 === 'D') break;
            if (t2 === 'B') {
                const bnx = nx + dx, bny = ny + dy;
                const bc2 = this.inBounds(bnx, bny) ? this.get(bnx, bny) : '#';
                if (bc2 === '#' || bc2 === 'D' || bc2 === 'B') break;
                this.grid[ny][nx] = '.';
                this.grid[bny][bnx] = 'B';
                if (this.getStatic(bnx, bny) === 'O') this.trigger('box_on_goal');
            }
            this.grid[ty][tx] = '.';
            this.grid[ny][nx] = 'A';
            tx = nx; ty = ny;

            if (this.getStatic(tx, ty) === 'P' && this.portalCD <= 0) {
                const dest = this.findPortalDest(tx, ty);
                if (dest && this.get(dest[0], dest[1]) !== '#' && this.get(dest[0], dest[1]) !== 'D' && this.get(dest[0], dest[1]) !== 'B') {
                    this.grid[ty][tx] = '.';
                    this.grid[dest[1]][dest[0]] = 'A';
                    this.trigger('portal_used');
                    this.portalCD = 3;
                    tx = dest[0]; ty = dest[1];
                    break;
                }
            }

            this.checkPos();
            if (this.events.hazard || this.events.goal_reached) break;
        }

        // Field effect on final position
        const sc = this.getStatic(tx, ty);
        if ('^v<>'.includes(sc)) {
            const [fdx, fdy] = FIELD_VEC[sc];
            const fnx = tx + fdx, fny = ty + fdy;
            if (this.inBounds(fnx, fny) && !('#D'.includes(this.get(fnx, fny)))) {
                if (this.get(fnx, fny) === 'B') {
                    const bnx = fnx + fdx, bny = fny + fdy;
                    if (this.inBounds(bnx, bny) && !('#DB'.includes(this.get(bnx, bny)))) {
                        this.grid[fny][fnx] = '.';
                        this.grid[bny][bnx] = 'B';
                    } else { this.checkPos(); return; }
                }
                this.grid[ty][tx] = '.';
                this.grid[fny][fnx] = 'A';
                tx = fnx; ty = fny;
            }
        }

        this.checkPos();
    }

    processGlobalFields() {
        // Move orbs on directional fields
        for (let y = 0; y < GRID_SIZE; y++) {
            for (let x = 0; x < GRID_SIZE; x++) {
                const sc = this.sgrid[y][x];
                if ('^v<>'.includes(sc)) {
                    const dyn = this.grid[y][x];
                    if (dyn === 'O') {
                        const [dx, dy] = FIELD_VEC[sc];
                        const nx = x + dx, ny = y + dy;
                        if (this.inBounds(nx, ny) && !('#DB'.includes(this.get(nx, ny))) && this.grid[ny][nx] === '.') {
                            this.grid[y][x] = '.';
                            this.grid[ny][nx] = 'O';
                        }
                    }
                }
            }
        }
    }

    run(actions, horizon = 0) {
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
        else if (this.stepCount >= steps) terminal = 'timeout';
        else terminal = 'timeout';

        // Timeline
        const effH = Math.max(horizon, steps, 1);
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

        return { finalGrid, events: { ...this.events }, eventTimeline: timeline, eventOrder: order.slice(0, 3), terminal };
    }
}

async function main() {
    const stream = fs.createReadStream('train.jsonl');
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

    let totalProbes = 0;
    let correctAgent = 0;
    let correctEvents = 0;
    let correctTerminal = 0;
    let count = 0;

    for await (const line of rl) {
        if (count >= 100) break;
        const d = JSON.parse(line);
        const initGrid = d.initial_full_grid;

        for (const ep of d.context_episodes) {
            totalProbes++;
            const sim = new Simulator(initGrid);
            const result = sim.run(ep.actions);

            // Compare agent position
            const predA = sim.findAgent();
            let trueA = null;
            for (let y = 0; y < 12; y++) {
                const x = ep.observed_final_full_grid[y].indexOf('A');
                if (x >= 0) { trueA = [x, y]; break; }
            }

            if (predA && trueA && predA[0] === trueA[0] && predA[1] === trueA[1]) {
                correctAgent++;
            }

            // Compare events
            const trueEvents = ep.observed_final_events;
            let eventMatch = true;
            for (const k of EVENT_KEYS) {
                if (result.events[k] !== trueEvents[k]) eventMatch = false;
            }
            if (eventMatch) correctEvents++;

            // Compare terminal
            if (result.terminal === ep.observed_final_terminal) correctTerminal++;
        }

        count++;
        if (count % 20 === 0) {
            console.log(`  Progress: ${count} contexts, ${totalProbes} probes`);
            console.log(`    Agent: ${correctAgent}/${totalProbes} = ${(100*correctAgent/totalProbes).toFixed(1)}%`);
            console.log(`    Events: ${correctEvents}/${totalProbes} = ${(100*correctEvents/totalProbes).toFixed(1)}%`);
            console.log(`    Terminal: ${correctTerminal}/${totalProbes} = ${(100*correctTerminal/totalProbes).toFixed(1)}%`);
        }
    }

    console.log(`\n=== Final Results (${totalProbes} probes) ===`);
    console.log(`Agent position: ${correctAgent}/${totalProbes} = ${(100*correctAgent/totalProbes).toFixed(1)}%`);
    console.log(`Events exact: ${correctEvents}/${totalProbes} = ${(100*correctEvents/totalProbes).toFixed(1)}%`);
    console.log(`Terminal: ${correctTerminal}/${totalProbes} = ${(100*correctTerminal/totalProbes).toFixed(1)}%`);
}

main().catch(console.error);
