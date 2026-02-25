import type {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { distributedRendererFactory } from './mimeRenderer';

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

// Export the renderer factory for MIME registration
export const rendererFactory = distributedRendererFactory;

export default [mimePlugin];
