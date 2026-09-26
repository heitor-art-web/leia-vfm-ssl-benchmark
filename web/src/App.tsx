import { useEffect, useMemo, useState } from 'react';
import navigationJson from './data/navigation.json';
import pagesJson from './data/pages.json';
import casesJson from './data/cases.json';
import methodsJson from './data/methods.json';
import policyJson from './data/annotation-policy.json';
import benchmarkJson from './data/benchmark.json';
import pipelineJson from './data/pipeline.json';
import type {
  AnnotationPolicyRow,
  BenchmarkData,
  CaseRecord,
  MethodRecord,
  NavigationItem,
  PageKey,
  PipelineNode,
} from './types';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Overview } from './pages/Overview';
import { Cases } from './pages/Cases';
import { Viewer } from './pages/Viewer';
import { Benchmark } from './pages/Benchmark';
import { Results } from './pages/Results';
import { About } from './pages/About';

const navigation = navigationJson as NavigationItem[];
const cases = casesJson as CaseRecord[];
const methods = methodsJson as MethodRecord[];
const policy = policyJson as AnnotationPolicyRow[];
const benchmark = benchmarkJson as BenchmarkData;
const pipeline = pipelineJson as PipelineNode[];
const pages = pagesJson as Record<PageKey, { title: string; subtitle: string }>;

function pageFromHash(): PageKey {
  const candidate = window.location.hash.replace('#/', '') as PageKey;
  return navigation.some((item) => item.id === candidate) ? candidate : 'overview';
}

export default function App() {
  const [page, setPage] = useState<PageKey>(pageFromHash);
  const [selectedCaseId, setSelectedCaseId] = useState('LIDC-IDRI-0191_4646251928');

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const selectedCase = useMemo(
    () => cases.find((record) => record.caseId === selectedCaseId) ?? cases[0],
    [selectedCaseId],
  );

  const navigate = (next: PageKey) => {
    window.location.hash = `/${next}`;
    setPage(next);
  };

  const openCase = (record: CaseRecord) => {
    setSelectedCaseId(record.caseId);
    navigate('viewer');
  };

  let content;
  if (page === 'overview') content = <Overview methods={methods} policy={policy} pipeline={pipeline} onNavigate={navigate} />;
  else if (page === 'cases') content = <Cases cases={cases} onOpen={openCase} />;
  else if (page === 'viewer') content = <Viewer cases={cases} selected={selectedCase} onSelect={(record) => setSelectedCaseId(record.caseId)} />;
  else if (page === 'benchmark') content = <Benchmark data={benchmark} methods={methods} />;
  else if (page === 'results') content = <Results data={benchmark} />;
  else content = <About />;

  return (
    <div className="app-shell">
      <Sidebar items={navigation} active={page} onNavigate={navigate} />
      <main className="main-shell">
        <Header title={pages[page].title} subtitle={pages[page].subtitle} />
        {content}
        <footer>
          <span>LIDC-IDRI pulmonary benchmark · research preview</span>
          <span>Green = trusted · Magenta = UNKNOWN</span>
        </footer>
      </main>
    </div>
  );
}
