/**
 * 分析训练数据中的物理机制
 * 目标：理解 ice、directional tiles、portals 的确切行为
 */
const fs = require('fs');
const readline = require('readline');

const GRID_SIZE = 12;
const DIRS = { U: [0, -1], D: [0, 1], L: [-1, 0], R: [1, 0] };
const REV = { U: 'D', D: 'U', L: 'R', R: 'L' };

async function main() {
    const samples = [];
    const stream = fs.createReadStream('train.jsonl');
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
    let count = 0;

    for await (const line of rl) {
        if (count >= 200) break;
        const d = JSON.parse(line);

        // 分析每个 probe episode
        for (let epIdx = 0; epIdx < d.context_episodes.length; epIdx++) {
            const ep = d.context_episodes[epIdx];
            const initA = d.initial_entities.A.split(',').map(Number);
            const initGrid = d.initial_full_grid.map(r => r.split(''));

            // 找到终局的 agent 位置
            let finalA = null;
            for (let y = 0; y < GRID_SIZE; y++) {
                const x = ep.observed_final_full_grid[y].indexOf('A');
                if (x >= 0) { finalA = [x, y]; break; }
            }

            // 计算动作预期位移 vs 实际位移
            let expectedX = initA[0], expectedY = initA[1];
            let agentDir = 'D'; // 默认方向
            let trace = [];

            for (let i = 0; i < ep.actions.length; i++) {
                const action = ep.actions[i];
                const obs = ep.observations[i]; // 执行前的观测

                // 从 sensor 读取方向
                if (obs && obs.sensor && obs.sensor.agent_dir) {
                    agentDir = obs.sensor.agent_dir;
                }

                trace.push({ step: i, action, dir: agentDir, pos: [expectedX, expectedY] });

                if (action === 'WAIT') continue;

                const [dx, dy] = DIRS[action] || [0, 0];
                const nx = expectedX + dx;
                const ny = expectedY + dy;

                if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) continue;
                const target = initGrid[ny][nx];

                // 简单追踪（忽略 ice/portal/fields 等）
                if (target === '#') continue;
                if (target === 'D') continue; // 门锁住
                if (target === 'B') {
                    const bnx = nx + dx, bny = ny + dy;
                    if (bnx >= 0 && bnx < GRID_SIZE && bny >= 0 && bny < GRID_SIZE &&
                        initGrid[bny][bnx] !== '#' && initGrid[bny][bnx] !== 'D' && initGrid[bny][bnx] !== 'B') {
                        initGrid[bny][bnx] = 'B';
                        initGrid[ny][nx] = '.';
                        expectedX = nx; expectedY = ny;
                    }
                } else {
                    expectedX = nx; expectedY = ny;
                }
            }

            const actualDx = finalA ? finalA[0] - initA[0] : null;
            const actualDy = finalA ? finalA[1] - initA[1] : null;
            const simpleDx = expectedX - initA[0];
            const simpleDy = expectedY - initA[1];

            if (actualDx !== simpleDx || actualDy !== simpleDy) {
                // 找出网格中所有 I、P、field 的位置
                const grid = d.initial_full_grid;
                const icePositions = [];
                const fieldPositions = [];
                const portalPositions = [];
                for (let y = 0; y < GRID_SIZE; y++) {
                    for (let x = 0; x < GRID_SIZE; x++) {
                        if (grid[y][x] === 'I') icePositions.push([x, y]);
                        if ('^v<>'.includes(grid[y][x])) fieldPositions.push([x, y, grid[y][x]]);
                        if (grid[y][x] === 'P') portalPositions.push([x, y]);
                    }
                }

                samples.push({
                    ctx: count, id: d.id, ep: epIdx,
                    initA, finalA,
                    simple: [simpleDx, simpleDy],
                    actual: [actualDx, actualDy],
                    actions: ep.actions.length,
                    terminal: ep.observed_final_terminal,
                    iceCount: icePositions.length,
                    fieldCount: fieldPositions.length,
                    portalCount: portalPositions.length,
                    iceDist: icePositions.map(p => Math.abs(p[0]-initA[0]) + Math.abs(p[1]-initA[1])),
                });
                if (samples.length >= 30) break;
            }
        }
        if (samples.length >= 30) break;
        count++;
    }

    console.log('=== 简单物理不足以解释的样本 ===');
    samples.forEach(s => {
        console.log(`ctx${s.ctx} ep${s.ep}: initA=${s.initA} final=${s.finalA} simple=(${s.simple[0]},${s.simple[1]}) actual=(${s.actual[0]},${s.actual[1]}) term=${s.terminal} I=${s.iceCount} F=${s.fieldCount} P=${s.portalCount}`);
    });
}

main().catch(console.error);
