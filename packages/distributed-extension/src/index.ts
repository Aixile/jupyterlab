import type {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ILayoutRestorer } from '@jupyterlab/application';
import { ITranslator } from '@jupyterlab/translation';
import { distributedRendererFactory } from './mimeRenderer';
import { DistributedStatusPanel } from './panel';

/**
 * Plugin that registers the distributed per-rank MIME renderer.
 */
const mimePlugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyterlab/distributed-extension:mime-renderer',
  description: 'Renders per-rank distributed output with rank selector.',
  autoStart: true,
  activate: (_app: JupyterFrontEnd): void => {
    // MIME renderer registration happens via the rendererFactory export
    console.warn('Distributed MIME renderer plugin activated');
  }
};

/**
 * Plugin that provides the distributed cluster status sidebar panel.
 */
const sidebarPlugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyterlab/distributed-extension:sidebar',
  description: 'Provides distributed cluster status sidebar.',
  autoStart: true,
  requires: [ITranslator],
  optional: [ILayoutRestorer],
  activate: (
    app: JupyterFrontEnd,
    translator: ITranslator,
    restorer: ILayoutRestorer | null
  ): void => {
    const trans = translator.load('jupyterlab');
    const panel = new DistributedStatusPanel();
    panel.title.caption = trans.__('Distributed Cluster');
    if (restorer) {
      restorer.add(panel, 'distributed-status');
    }
    app.shell.add(panel, 'left', { rank: 300, type: 'Distributed' });
  }
};

// Export the renderer factory for MIME registration
export const rendererFactory = distributedRendererFactory;

export default [mimePlugin, sidebarPlugin];
