export const DISTRIBUTED_MIME =
  'application/vnd.jupyterlab.distributed+json';

export interface IRankOutput {
  outputs: IWorkerOutput[];
  status: 'ok' | 'error' | 'timeout';
  execution_time: number;
}

export interface IWorkerOutput {
  type: 'stream' | 'display_data' | 'execute_result' | 'error';
  msg_id: string;
  name?: string;
  text?: string;
  data?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  ename?: string;
  evalue?: string;
  traceback?: string[];
}

export interface IRankOutputsMessage {
  type: 'rank_outputs';
  msg_id?: string;
  ranks: Record<string, IRankOutput>;
}

export interface IClusterStatusMessage {
  type: 'cluster_status';
  registered: number;
  expected: number;
  nodes: Record<string, INodeStatus>;
  gpu_memory_total_mb: number;
  gpu_memory_used_mb: number;
}

export interface INodeStatus {
  ranks: number[];
  status: 'healthy' | 'degraded' | 'disconnected';
}

export type IDistributedMessage = IRankOutputsMessage | IClusterStatusMessage;
