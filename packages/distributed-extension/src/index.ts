import type {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@jupyterlab/distributed-extension:plugin',
  description: 'Provides distributed SPMD cell execution support.',
  autoStart: true,
  activate: (app: JupyterFrontEnd): void => {
    console.warn('Distributed extension activated');
  }
};

export default [plugin];
