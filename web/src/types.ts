export type PageKey = 'overview' | 'cases' | 'viewer' | 'benchmark' | 'results' | 'about';

export interface NavigationItem {
  id: PageKey;
  label: string;
  glyph: string;
}

export interface CaseRecord {
  patientId: string;
  caseId: string;
  role: 'control' | 'trusted' | 'unknown';
  title: string;
  description: string;
  annotationVotes: number;
  diameterMm: number | null;
  malignancyMedian: number | null;
  assets: {
    contactSheet: string;
  };
}

export interface MethodRecord {
  id: 'sup' | 'mt' | 'mt-medsam';
  name: string;
  model: string;
  status: 'implemented' | 'planned';
  description: string;
}

export interface PipelineNode {
  id: string;
  title: string;
  subtitle: string;
}

export interface AnnotationPolicyRow {
  votes: string;
  label: string;
  value: number;
  tone: 'background' | 'trusted' | 'unknown';
}

export interface BenchmarkData {
  split: {
    train: number;
    validation: number;
    test: number;
  };
  budgets: Array<{ id: string; label: string; patients: number }>;
  seeds: number[];
  metrics: string[];
  matrix: Array<{
    method: string;
    budget: string;
    status: 'pending' | 'smoke-only' | 'complete';
    dice: number | null;
    iou: number | null;
  }>;
}
