import { Token } from '@lumino/coreutils';
import type { ISignal } from '@lumino/signaling';
import type { IClusterStatusMessage } from './types';

export interface IDistributedStatus {
  readonly statusChanged: ISignal<IDistributedStatus, IClusterStatusMessage>;
  readonly isDistributed: boolean;
  readonly clusterStatus: IClusterStatusMessage | null;
}

export const IDistributedStatus = new Token<IDistributedStatus>(
  '@jupyterlab/distributed:IDistributedStatus',
  'Provides distributed cluster status.'
);
