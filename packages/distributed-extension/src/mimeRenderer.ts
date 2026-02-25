import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Widget } from '@lumino/widgets';
import {
  DISTRIBUTED_MIME,
  type IRankOutputsMessage,
  type IRankOutput
} from '@jupyterlab/distributed';

/**
 * A renderer widget for distributed per-rank outputs.
 *
 * Displays a rank selector tab bar and the corresponding output
 * for each rank. Supports single-rank view and an "All Ranks"
 * collapsible view.
 */
class DistributedOutputRenderer extends Widget implements IRenderMime.IRenderer {
  private _selectedRank = 0;
  private _ranksData: Record<string, IRankOutput> = {};

  constructor() {
    super();
    this.addClass('jp-DistributedOutput');
  }

  async renderModel(model: IRenderMime.IMimeModel): Promise<void> {
    const data = model.data[
      DISTRIBUTED_MIME
    ] as unknown as IRankOutputsMessage;
    if (!data || data.type !== 'rank_outputs') {
      return;
    }
    this._ranksData = data.ranks;
    this._render();
  }

  private _render(): void {
    // Clear and rebuild
    this.node.innerHTML = '';
    const ranks = Object.keys(this._ranksData)
      .map(Number)
      .sort((a, b) => a - b);
    if (ranks.length === 0) {
      return;
    }

    // Tab bar with rank buttons
    const tabBar = document.createElement('div');
    tabBar.className = 'jp-DistributedOutput-tabBar';
    for (const rank of ranks) {
      const tab = document.createElement('button');
      tab.className = 'jp-DistributedOutput-tab';
      tab.textContent = `${rank}`;
      const rankData = this._ranksData[String(rank)];
      if (rankData.status === 'error') {
        tab.classList.add('jp-mod-error');
      } else if (rankData.status === 'ok') {
        tab.classList.add('jp-mod-ok');
      }
      if (rank === this._selectedRank) {
        tab.classList.add('jp-mod-active');
      }
      tab.addEventListener('click', () => {
        this._selectedRank = rank;
        this._render();
      });
      tabBar.appendChild(tab);
    }
    // "All Ranks" button
    const allBtn = document.createElement('button');
    allBtn.className =
      'jp-DistributedOutput-tab jp-DistributedOutput-allTab';
    allBtn.textContent = 'All Ranks';
    allBtn.addEventListener('click', () => {
      this._selectedRank = -1;
      this._render();
    });
    if (this._selectedRank === -1) {
      allBtn.classList.add('jp-mod-active');
    }
    tabBar.appendChild(allBtn);
    this.node.appendChild(tabBar);

    // Output content
    const outputArea = document.createElement('div');
    outputArea.className = 'jp-DistributedOutput-content';
    if (this._selectedRank === -1) {
      for (const rank of ranks) {
        outputArea.appendChild(this._renderRankSection(rank));
      }
    } else {
      outputArea.appendChild(this._renderRankOutputs(this._selectedRank));
    }
    this.node.appendChild(outputArea);
  }

  private _renderRankSection(rank: number): HTMLElement {
    const section = document.createElement('details');
    section.className = 'jp-DistributedOutput-rankSection';
    const summary = document.createElement('summary');
    const rankData = this._ranksData[String(rank)];
    const icon = rankData.status === 'ok' ? '\u2713' : '\u2717';
    summary.textContent = `Rank ${rank} ${icon} (${rankData.execution_time.toFixed(2)}s)`;
    section.appendChild(summary);
    section.appendChild(this._renderRankOutputs(rank));
    return section;
  }

  private _renderRankOutputs(rank: number): HTMLElement {
    const container = document.createElement('div');
    container.className = 'jp-DistributedOutput-rankContent';
    const rankData = this._ranksData[String(rank)];
    if (!rankData) {
      container.textContent = `No output for rank ${rank}`;
      return container;
    }
    for (const output of rankData.outputs) {
      const el = document.createElement('pre');
      el.className = 'jp-DistributedOutput-outputItem';
      if (output.type === 'stream') {
        el.textContent = output.text || '';
        if (output.name === 'stderr') {
          el.classList.add('jp-mod-stderr');
        }
      } else if (output.type === 'error') {
        el.textContent = (output.traceback || []).join('\n');
        el.classList.add('jp-mod-error');
      } else if (
        output.type === 'display_data' ||
        output.type === 'execute_result'
      ) {
        const textData = output.data?.['text/plain'];
        el.textContent =
          typeof textData === 'string'
            ? textData
            : JSON.stringify(output.data);
      }
      container.appendChild(el);
    }
    if (rankData.outputs.length === 0) {
      container.textContent = '(no output)';
    }
    return container;
  }
}

/**
 * A renderer factory for distributed per-rank outputs.
 */
export const distributedRendererFactory: IRenderMime.IRendererFactory = {
  safe: true,
  mimeTypes: [DISTRIBUTED_MIME],
  defaultRank: 80,
  createRenderer: (_options: IRenderMime.IRendererOptions) =>
    new DistributedOutputRenderer()
};
