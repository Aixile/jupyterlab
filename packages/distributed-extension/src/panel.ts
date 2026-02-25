import { Widget } from '@lumino/widgets';
import type { IClusterStatusMessage } from '@jupyterlab/distributed';

/**
 * A sidebar panel that displays real-time distributed cluster status.
 *
 * Shows worker registration progress, per-node rank details, and
 * GPU memory utilization when available.
 */
export class DistributedStatusPanel extends Widget {
  private _status: IClusterStatusMessage | null = null;

  constructor() {
    super();
    this.id = 'jp-distributed-status';
    this.addClass('jp-DistributedStatusPanel');
    this._render();
  }

  /**
   * Update the panel with new cluster status data and re-render.
   */
  updateStatus(status: IClusterStatusMessage): void {
    this._status = status;
    this._render();
  }

  private _render(): void {
    this.node.innerHTML = '';
    if (!this._status) {
      this.node.innerHTML =
        '<div class="jp-DistributedStatusPanel-empty">No distributed session active.</div>';
      return;
    }
    const s = this._status;

    // Header with worker count
    const header = document.createElement('div');
    header.className = 'jp-DistributedStatusPanel-header';
    header.innerHTML = `<div class="jp-DistributedStatusPanel-row"><span>Workers:</span><strong>${s.registered}/${s.expected}</strong></div>`;
    this.node.appendChild(header);

    // Progress bar
    const progress = document.createElement('div');
    progress.className = 'jp-DistributedStatusPanel-progress';
    const pct = s.expected > 0 ? (s.registered / s.expected) * 100 : 0;
    progress.innerHTML = `<div class="jp-DistributedStatusPanel-progressBar" style="width: ${pct}%"></div>`;
    this.node.appendChild(progress);

    // Nodes tree
    const nodesSection = document.createElement('div');
    nodesSection.className = 'jp-DistributedStatusPanel-nodes';
    for (const [hostname, nodeStatus] of Object.entries(s.nodes)) {
      const nodeEl = document.createElement('details');
      nodeEl.open = true;
      const summary = document.createElement('summary');
      const dot = nodeStatus.status === 'healthy' ? '\u25CF' : '\u25CB';
      summary.textContent = `${dot} ${hostname} (${nodeStatus.ranks.length} ranks)`;
      nodeEl.appendChild(summary);
      const rankList = document.createElement('div');
      rankList.className = 'jp-DistributedStatusPanel-rankList';
      for (const rank of nodeStatus.ranks) {
        const rankEl = document.createElement('span');
        rankEl.className = 'jp-DistributedStatusPanel-rank';
        rankEl.textContent = `r${rank}`;
        rankList.appendChild(rankEl);
      }
      nodeEl.appendChild(rankList);
      nodesSection.appendChild(nodeEl);
    }
    this.node.appendChild(nodesSection);

    // GPU memory
    if (s.gpu_memory_total_mb > 0) {
      const mem = document.createElement('div');
      mem.className = 'jp-DistributedStatusPanel-memory';
      const usedGB = (s.gpu_memory_used_mb / 1024).toFixed(1);
      const totalGB = (s.gpu_memory_total_mb / 1024).toFixed(1);
      const memPct =
        (s.gpu_memory_used_mb / s.gpu_memory_total_mb) * 100;
      mem.innerHTML =
        `<div>GPU Memory: ${usedGB}/${totalGB} GB</div>` +
        `<div class="jp-DistributedStatusPanel-progress">` +
        `<div class="jp-DistributedStatusPanel-progressBar ${memPct > 90 ? 'jp-mod-warning' : ''}" style="width: ${memPct}%"></div>` +
        `</div>`;
      this.node.appendChild(mem);
    }
  }
}
