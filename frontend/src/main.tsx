import { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Layers,
  Workflow,
  Database,
  ArrowUpRight,
  ChevronDown,
  ChevronRight,
  Play,
  ShieldCheck,
  RotateCcw,
  Radio,
  CheckCircle2,
  AlertTriangle,
  Clock3,
  Target,
  Terminal,
  BarChart3,
  KeyRound,
  X,
  RefreshCw,
  Menu,
} from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";
import "./style.css";

type Metrics = {
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
  brier: number;
  samples: number;
  failure_rate: number;
  confusion_matrix: number[][];
};
type Model = {
  id: string;
  kind: string;
  dataset: string;
  run_id: string;
  registry_version: string;
  metrics: Metrics;
  validation: Metrics;
  threshold: number;
  gate: { passed: boolean; reasons: string[] };
};
type Feature = {
  name: string;
  psi: number;
  drifted: boolean;
  reference_mean: number;
  current_mean: number;
  p_value: number;
  distribution: { reference: number[]; current: number[]; edges: number[] };
};
type Drift = {
  id: string;
  kind: string;
  drift_detected: boolean;
  features: Feature[];
  created: number;
  evidently_html: string | null;
  reference_dataset: string;
  current_dataset: string;
  prediction_drift?: { psi: number };
};
type Dataset = {
  id: string;
  rows: number;
  failure_rate: number;
  created: number;
  provenance: { shifted?: boolean; seed?: number };
};
type Overview = {
  rollouts: {id:string;candidate:string;champion:string;status:string;fraction:number;decision:{reasons:string[]}|null;checks:{passed:boolean;ready:boolean;candidate_requests:number;error_rate:number;p95_ms:number|null;reasons:string[];waiting_for:string|null}}[];
  proposals: {id:string;dataset:string;status:string;reason:string[];models:string[]}[];
  champion: string | null;
  previous: string | null;
  models: Model[];
  predictions: number;
  p95_ms: number | null;
  recent_predictions: {
    model_id: string;
    risk: number;
    predicted: number;
    label: number | null;
    latency_ms: number;
    created: number;
  }[];
  datasets: Dataset[];
  reports: Drift[];
  audit: {
    action: string;
    payload: Record<string, unknown>;
    created: number;
  }[];
  pipeline: { status: string; error: string | null };
  label_performance: {
    status: string;
    labeled_samples?: number;
    metrics?: Metrics;
  } | null;
};
type Prediction = {
  prediction_id: string;
  risk: number;
  failure_predicted: boolean;
  threshold: number;
  registry_version: string;
  latency_ms: number;
};
const pretty = (value: string) => value.replace(/[_-]/g, " ");
const number = (n: number | null | undefined, digits = 2) =>
  n == null ? "—" : n.toFixed(digits);
async function request<T>(
  path: string,
  body?: unknown,
  token?: string,
): Promise<T> {
  const response = await fetch("/api" + path, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      ...(token ? { "X-Admin-Token": token } : {}),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) {
    let message = "API unavailable";
    try {
      const data = await response.json();
      message =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail);
    } catch {}
    throw new Error(message);
  }
  return response.json();
}

function App() {
  const [data, setData] = useState<Overview | null>(null),
    [view, setView] = useState("overview"),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [token, setToken] = useState(""),
    [showToken, setShowToken] = useState(false),
    [busy, setBusy] = useState(false),
    [showPipeline, setShowPipeline] = useState(false),
    [dataset, setDataset] = useState(""),
    [mobile, setMobile] = useState(false),
    [reading, setReading] = useState({
      temperature: 82,
      vibration: 4.2,
      pressure: 5.3,
      rpm: 1850,
      operating_hours: 8200,
    }),
    [prediction, setPrediction] = useState<Prediction | null>(null),
    [feature, setFeature] = useState("temperature");
  async function refresh() {
    try {
      setData(await request<Overview>("/overview"));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, []);
  async function action(path: string, body: unknown = {}) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await request<Record<string, unknown>>(path, body, token);
      await refresh();
      setNotice(
        path === "/train"
          ? "Training queued. Review registered candidates before promotion."
          : "Action recorded in the audit trail.",
      );
      return result;
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function training() {
    const id = dataset || data?.datasets.find((d) => !d.provenance.shifted)?.id;
    if (!id) {
      setError("Prepare a dataset first.");
      return;
    }
    const approved = await action("/datasets/approve", { dataset_id: id });
    if (approved) {
      await action("/train");
      setShowPipeline(false);
    }
  }
  async function predict() {
    setBusy(true);
    setError("");
    try {
      setPrediction(await request<Prediction>("/predict", reading));
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const champion = data?.models.find((m) => m.id === data.champion),
    latest = data?.reports.find((r) => r.kind === "drift"),
    selectedFeature = latest?.features.find((f) => f.name === feature),
    riskSeries = [...(data?.recent_predictions || [])]
      .reverse()
      .map((p, i) => ({ sample: i + 1, risk: Math.round(p.risk * 100) })),
    driftSeries =
      latest?.features.map((f) => ({
        name: pretty(f.name),
        psi: Number(f.psi.toFixed(3)),
      })) || [];
  const routes = [
    ["overview", "Overview", Activity],
    ["registry", "Model registry", Layers],
    ["monitoring", "Monitoring", Radio],
    ["inference", "Inference lab", Terminal],
    ["lifecycle", "Release control", ShieldCheck],
  ] as const;
  return (
    <div className="app">
      <aside className={"sidebar " + (mobile ? "open" : "")}>
        <a className="brand" href="#">
          <span className="brand-icon">
            <Workflow size={23} />
          </span>
          ForgeOps<span className="brand-dot">.</span>
        </a>
        <div className="workspace">
          <span>PM</span>
          <div>
            Predictive maintenance<small>Industrial sensor pipeline</small>
          </div>
          <ChevronDown size={13} />
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {routes.map(([key, label, Icon]) => (
            <button
              key={key}
              className={view === key ? "active" : ""}
              onClick={() => {
                setView(key);
                setMobile(false);
              }}
            >
              <Icon size={17} />
              {label}
              {key === "monitoring" && latest?.drift_detected && (
                <span className="nav-dot" />
              )}
            </button>
          ))}
        </nav>
        <div className="nav-label">SERVICE LINKS</div>
        <a
          className="service"
          href="http://127.0.0.1:5001"
          target="_blank"
          rel="noreferrer"
        >
          <Database size={16} /> MLflow experiments <ArrowUpRight size={13} />
        </a>
        <a
          className="service"
          href="http://127.0.0.1:8200/docs"
          target="_blank"
          rel="noreferrer"
        >
          <Terminal size={16} /> API documentation <ArrowUpRight size={13} />
        </a>
        <a
          className="service"
          href="http://127.0.0.1:9091"
          target="_blank"
          rel="noreferrer"
        >
          <BarChart3 size={16} /> Prometheus <ArrowUpRight size={13} />
        </a>
        <div className="sidebar-bottom">
          <div className="environment">
            <span className="live-dot" /> LOCAL ENVIRONMENT
            <small>Simulated data · No real equipment</small>
          </div>
          <div className="user">
            <span>MF</span>
            <div>
              Engineering workspace<small>Controlled model lifecycle</small>
            </div>
          </div>
        </div>
      </aside>
      <main>
        <header>
          <button
            className="mobile"
            aria-label="Toggle menu"
            onClick={() => setMobile(!mobile)}
          >
            <Menu size={19} />
          </button>
          <div className="breadcrumbs">
            Workspace <ChevronRight size={13} />{" "}
            <strong>{routes.find((r) => r[0] === view)?.[1]}</strong>
          </div>
          <div className="header-actions">
            <span className="connected">
              <span className="live-dot" />{" "}
              {data ? "API connected" : "Connecting"}
            </span>
            <button
              aria-label="Administrator access"
              title="Administrator access"
              onClick={() => setShowToken(!showToken)}
            >
              <KeyRound size={17} />
            </button>
            <button aria-label="Refresh data" onClick={refresh}>
              <RefreshCw size={16} />
            </button>
          </div>
        </header>
        <div className="content">
          {showToken && (
            <div className="token-panel">
              <KeyRound size={17} />
              <label>
                Administrator token
                <input
                  type="password"
                  autoComplete="off"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="Required for training, promotion, and monitoring"
                />
              </label>
              <small>
                Held in memory only. Read the local runtime/admin-token.txt
                file.
              </small>
            </div>
          )}
          <div className="page-title">
            <div>
              <div className="eyebrow">MODEL OPERATIONS / SENSOR-V1</div>
              <h1>
                {view === "overview"
                  ? "From signal to certainty."
                  : view === "registry"
                    ? "Better models. Deliberate releases."
                    : view === "monitoring"
                      ? "Observe the shift. Verify the impact."
                      : view === "lifecycle" ? "Evidence before expansion." : "Put your model to the test."}
              </h1>
              <p>
                {view === "overview"
                  ? "A clear view of your predictive maintenance lifecycle."
                  : view === "registry"
                    ? "Held-out evidence, explicit approval, and a path back."
                    : view === "monitoring"
                      ? "Distribution changes propose a review. They never trigger a deployment."
                      : view === "lifecycle" ? "Drift proposals, approved retraining, and monitored canary releases." : "Score a sensor reading with the active deployment."}
              </p>
            </div>
            <button
              className="primary"
              disabled={busy}
              onClick={() => setShowPipeline(true)}
            >
              <Play size={14} /> Run pipeline
            </button>
          </div>
          {error && (
            <div className="alert error" role="alert">
              <AlertTriangle size={17} />
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={15} />
              </button>
            </div>
          )}
          {notice && (
            <div className="alert notice" role="status">
              <CheckCircle2 size={17} />
              {notice}
              <button onClick={() => setNotice("")} aria-label="Dismiss notice">
                <X size={15} />
              </button>
            </div>
          )}
          {!data ? (
            <div className="empty">
              Connecting to ForgeOps. Start the FastAPI server on port 8200.
            </div>
          ) : (
            <>
              {view === "lifecycle" && <>
                <section className="panel"><div className="panel-head"><h2>Retraining proposals</h2><span className="pill">Approval required</span></div>
                <p className="subtle">Monitoring creates a proposal when at least two features drift or delayed-label quality degrades. Approve its dataset before training.</p>
                {!data.proposals.length && <div className="empty">Run a monitoring batch to evaluate retraining criteria.</div>}
                {data.proposals.map(p => <div className="release-card" key={p.id}><div><strong>{p.status.toUpperCase()}</strong><p>{p.reason.join(" · ")}</p><small>Dataset {p.dataset.slice(0,12)} · {p.models.length} registered challengers</small></div>
                {p.status === "pending" && <div className="release-actions"><button disabled={busy} onClick={() => action("/datasets/approve",{dataset_id:p.dataset})}>Approve dataset</button><button className="primary" disabled={busy} onClick={() => action(`/retraining/${p.id}/train`)}>Train challengers</button></div>}</div>)}
                </section>
                <section className="panel"><div className="panel-head"><h2>Canary deployments</h2><span className="pill">80% champion / 20% challenger</span></div>
                <p className="subtle">Start a release in Model registry. Requests route deterministically. Error, latency and labeled quality checks can disable the candidate automatically.</p>
                {!data.rollouts.length && <div className="empty">No canary releases yet. The initial champion is serving all traffic.</div>}
                {data.rollouts.map(r => <div className="release-card" key={r.id}><div><strong>{r.status.toUpperCase()} · {r.id.slice(0,8)}</strong><p>Candidate {r.candidate.slice(0,8)} · champion {r.champion.slice(0,8)}</p>
                <p>{r.checks.candidate_requests} candidate requests · {(r.checks.error_rate*100).toFixed(1)}% errors · p95 {number(r.checks.p95_ms)} ms</p><small>{r.decision?.reasons?.join(" · ") || r.checks.reasons.join(" · ") || r.checks.waiting_for || "All rollout checks passed"}</small></div>
                {r.status === "active" && <div className="release-actions"><button disabled={busy} onClick={() => action(`/rollouts/${r.id}/abort`)}>Roll back canary</button><button className="primary" disabled={busy || !r.checks.passed} onClick={() => action(`/rollouts/${r.id}/complete`)}>Approve full rollout</button></div>}</div>)}
                </section>
              </>}
              {view === "overview" && (
                <>
                  <div className="summary-grid">
                    <Metric
                      label="CHAMPION PR-AUC"
                      value={number(champion?.metrics.pr_auc, 3)}
                      note={
                        champion
                          ? `v${champion.registry_version} · ${pretty(champion.kind)}`
                          : "Awaiting an approved model"
                      }
                      icon={<Target size={17} />}
                      accent
                    />
                    <Metric
                      label="PREDICTIONS SERVED"
                      value={data.predictions.toLocaleString()}
                      note="Logged with model lineage"
                      icon={<Activity size={17} />}
                    />
                    <Metric
                      label="P95 COMPUTE LATENCY"
                      value={
                        data.p95_ms == null
                          ? "—"
                          : number(data.p95_ms, 1) + " ms"
                      }
                      note="Recent 500 calls · excludes HTTP"
                      icon={<Clock3 size={17} />}
                    />
                    <Metric
                      label="DRIFTED FEATURES"
                      value={
                        latest
                          ? `${latest.features.filter((f) => f.drifted).length} / ${latest.features.length}`
                          : "—"
                      }
                      note={
                        latest
                          ? "Latest batch comparison"
                          : "Run a monitoring batch"
                      }
                      icon={<Radio size={17} />}
                    />
                  </div>
                  <div className="lifecycle">
                    <div className="panel-heading">
                      <h2>The lifecycle, at a glance</h2>
                      <span className="pill">{data.pipeline.status}</span>
                    </div>
                    <div className="steps">
                      {[
                        [
                          "Validate",
                          "Pandera contract",
                          data.datasets.length > 0,
                        ],
                        [
                          "Train",
                          "Baseline + candidates",
                          data.models.length > 0,
                        ],
                        [
                          "Evaluate",
                          "Held-out quality gates",
                          data.models.length > 0,
                        ],
                        ["Register", "MLflow versions", data.models.length > 0],
                        ["Deploy", "Explicit approval", !!champion],
                        ["Monitor", "Drift + delayed labels", !!latest],
                      ].map(([title, caption, ready], i) => (
                        <div key={String(title)}>
                          <span className={ready ? "step done" : "step"}>
                            {ready ? <CheckCircle2 size={17} /> : i + 1}
                          </span>
                          <strong>{title}</strong>
                          <small>{caption}</small>
                        </div>
                      ))}
                    </div>
                  </div>
                  {latest?.drift_detected && (
                    <div className="drift-banner">
                      <span className="warning-icon">
                        <AlertTriangle size={20} />
                      </span>
                      <div>
                        <strong>
                          A distribution shift needs your attention
                        </strong>
                        <p>
                          {latest.features
                            .filter((f) => f.drifted)
                            .map((f) => pretty(f.name))
                            .join(", ")}{" "}
                          drifted in the latest batch. Review labels before
                          approving retraining.
                        </p>
                      </div>
                      <button onClick={() => setView("monitoring")}>
                        Inspect report <ArrowUpRight size={15} />
                      </button>
                    </div>
                  )}
                  <div className="chart-grid">
                    <section className="panel">
                      <div className="panel-heading">
                        <div>
                          <h2>Failure risk across recent readings</h2>
                          <p>Observed predictions · chronological order</p>
                        </div>
                        <span className="legend">
                          <i /> Risk %
                        </span>
                      </div>
                      <div className="chart">
                        {riskSeries.length ? (
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={riskSeries}>
                              <defs>
                                <linearGradient
                                  id="riskFill"
                                  x1="0"
                                  y1="0"
                                  x2="0"
                                  y2="1"
                                >
                                  <stop
                                    offset="0%"
                                    stopColor="#b76542"
                                    stopOpacity={0.22}
                                  />
                                  <stop
                                    offset="100%"
                                    stopColor="#b76542"
                                    stopOpacity={0}
                                  />
                                </linearGradient>
                              </defs>
                              <CartesianGrid
                                strokeDasharray="3 6"
                                vertical={false}
                                stroke="#e9e7e0"
                              />
                              <XAxis
                                dataKey="sample"
                                tick={{ fontSize: 10, fill: "#92958b" }}
                                axisLine={false}
                                tickLine={false}
                              />
                              <YAxis
                                domain={[0, 100]}
                                tick={{ fontSize: 10, fill: "#92958b" }}
                                axisLine={false}
                                tickLine={false}
                              />
                              <Tooltip />
                              <Area isAnimationActive={false}
                                type="monotone"
                                dataKey="risk"
                                stroke="#b76542"
                                strokeWidth={2}
                                fill="url(#riskFill)"
                              />
                            </AreaChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="empty">
                            Serve your first prediction to see the signal.
                          </div>
                        )}
                      </div>
                    </section>
                    <section className="panel">
                      <div className="panel-heading">
                        <div>
                          <h2>Input distribution drift</h2>
                          <p>Population Stability Index · latest batch</p>
                        </div>
                        <button
                          onClick={() => setView("monitoring")}
                          aria-label="Inspect drift"
                        >
                          <ArrowUpRight size={17} />
                        </button>
                      </div>
                      <div className="chart">
                        {driftSeries.length ? (
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={driftSeries}
                              layout="vertical"
                              margin={{ left: 10, right: 20 }}
                            >
                              <CartesianGrid
                                strokeDasharray="3 6"
                                horizontal={false}
                                stroke="#e9e7e0"
                              />
                              <XAxis
                                type="number"
                                tick={{ fontSize: 10 }}
                                axisLine={false}
                                tickLine={false}
                              />
                              <YAxis
                                dataKey="name"
                                type="category"
                                width={95}
                                tick={{ fontSize: 10, fill: "#777f70" }}
                                axisLine={false}
                                tickLine={false}
                              />
                              <Tooltip />
                              <Bar isAnimationActive={false}
                                dataKey="psi"
                                fill="#9ba78b"
                                radius={[0, 4, 4, 0]}
                                barSize={14}
                              />
                            </BarChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="empty">No monitoring report yet.</div>
                        )}
                      </div>
                    </section>
                  </div>
                  <Registry data={data} action={action} busy={busy} compact />
                  <div className="bottom-grid">
                    <section className="panel">
                      <div className="panel-heading">
                        <h2>Latest activity</h2>
                        <span className="subtle">Audit trail</span>
                      </div>
                      <div className="activity-list">
                        {data.audit.slice(0, 5).map((a, i) => (
                          <div key={i}>
                            <span className="event-dot" />
                            <div>
                              <strong>{pretty(a.action)}</strong>
                              <small>
                                {new Date(a.created * 1000).toLocaleString()}
                              </small>
                            </div>
                            <ShieldCheck size={14} />
                          </div>
                        ))}
                      </div>
                    </section>
                    <section className="panel label-panel">
                      <div className="panel-heading">
                        <h2>After the labels arrive</h2>
                        <span className="pill">
                          {data.label_performance?.status === "measured"
                            ? "Measured"
                            : "Waiting"}
                        </span>
                      </div>
                      {data.label_performance?.metrics ? (
                        <>
                          <div className="label-metrics">
                            <div>
                              <strong>
                                {number(
                                  data.label_performance.metrics.pr_auc,
                                  3,
                                )}
                              </strong>
                              <small>PR-AUC</small>
                            </div>
                            <div>
                              <strong>
                                {number(
                                  data.label_performance.metrics.recall,
                                  3,
                                )}
                              </strong>
                              <small>Recall</small>
                            </div>
                            <div>
                              <strong>
                                {data.label_performance.metrics.samples}
                              </strong>
                              <small>Labeled readings</small>
                            </div>
                          </div>
                          <p>
                            Realized performance on labels received so far.
                            Label selection may bias this estimate.
                          </p>
                        </>
                      ) : (
                        <p>
                          At least 30 labeled readings and both classes are
                          needed before reporting performance.
                        </p>
                      )}
                    </section>
                  </div>
                </>
              )}
              {view === "registry" && (
                <>
                  <div className="registry-intro">
                    <ShieldCheck size={24} />
                    <div>
                      <strong>Every release has to earn its place.</strong>
                      <p>
                        Minimum PR-AUC, recall, precision, calibration, schema
                        compatibility, and champion regression limits are
                        enforced on the server.
                      </p>
                    </div>
                    <button
                      className="secondary"
                      disabled={busy || !data.previous}
                      onClick={() => action("/models/rollback")}
                    >
                      <RotateCcw size={14} /> Approve rollback
                    </button>
                  </div>
                  <Registry data={data} action={action} busy={busy} />
                  <section className="panel rejection">
                    <div className="panel-heading">
                      <h2>A rejected model is a successful guardrail</h2>
                      <span className="pill warning">Fixed baseline</span>
                    </div>
                    <pre>{`$ mlops_platform promote BASELINE_ID --approve\n${data.models.find((m) => m.kind === "fixed-baseline")?.gate.reasons.join("\n") || "Train the baseline to inspect its gate result."}\n\nPromotion blocked. The champion remains unchanged.`}</pre>
                  </section>
                </>
              )}
              {view === "monitoring" && (
                <>
                  <div className="monitor-toolbar">
                    <h2>Batch monitoring</h2>
                    <button
                      className="secondary"
                      disabled={busy || data.datasets.length < 2}
                      onClick={() => {
                        const reference = data.datasets.find(
                            (d) => !d.provenance.shifted,
                          ),
                          current = data.datasets.find(
                            (d) => d.provenance.shifted,
                          );
                        if (reference && current)
                          action("/monitor", {
                            reference_id: reference.id,
                            current_id: current.id,
                          });
                        else
                          setError(
                            "Prepare reference and shifted datasets first.",
                          );
                      }}
                    >
                      <RefreshCw size={14} /> Run drift report
                    </button>
                  </div>
                  {latest ? (
                    <>
                      <div className="drift-banner">
                        <AlertTriangle size={21} />
                        <div>
                          <strong>
                            {latest.drift_detected
                              ? "Drift detected. Deployment unchanged."
                              : "No feature crossed the drift threshold."}
                          </strong>
                          <p>
                            PSI &gt; 0.2 and Bonferroni-corrected KS p-value
                            &lt; 0.01 / 5. Prediction PSI:{" "}
                            {number(latest.prediction_drift?.psi, 3)}.
                          </p>
                        </div>
                        {latest.evidently_html && (
                          <a
                            href={latest.evidently_html}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Evidently report <ArrowUpRight size={14} />
                          </a>
                        )}
                      </div>
                      <div className="feature-tabs">
                        {latest.features.map((f) => (
                          <button
                            className={feature === f.name ? "active" : ""}
                            key={f.name}
                            onClick={() => setFeature(f.name)}
                          >
                            {pretty(f.name)}{" "}
                            {f.drifted && <span className="nav-dot" />}
                          </button>
                        ))}
                      </div>
                      <section className="panel">
                        <div className="panel-heading">
                          <div>
                            <h2>{pretty(feature)} distribution</h2>
                            <p>
                              Reference and current batch · percentage of
                              readings
                            </p>
                          </div>
                          <span className="pill">
                            PSI {number(selectedFeature?.psi, 3)}
                          </span>
                        </div>
                        <div className="large-chart">
                          {selectedFeature && (
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart
                                data={selectedFeature.distribution.reference.map(
                                  (count, i) => ({
                                    bin: selectedFeature.distribution.edges[
                                      i
                                    ].toFixed(1),
                                    reference: count,
                                    current:
                                      selectedFeature.distribution.current[i],
                                  }),
                                )}
                              >
                                <CartesianGrid
                                  strokeDasharray="3 6"
                                  vertical={false}
                                />
                                <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                                <YAxis tick={{ fontSize: 10 }} />
                                <Tooltip />
                                <Bar isAnimationActive={false}
                                  dataKey="reference"
                                  fill="#a8b197"
                                  radius={[3, 3, 0, 0]}
                                />
                                <Bar isAnimationActive={false}
                                  dataKey="current"
                                  fill="#bd7756"
                                  radius={[3, 3, 0, 0]}
                                />
                              </BarChart>
                            </ResponsiveContainer>
                          )}
                        </div>
                        <div className="chart-legend">
                          <span>
                            <i style={{ background: "#a8b197" }} /> Reference
                          </span>
                          <span>
                            <i style={{ background: "#bd7756" }} /> Current
                          </span>
                        </div>
                      </section>
                      <div className="feature-results">
                        {latest.features.map((f) => (
                          <article key={f.name}>
                            <strong>{pretty(f.name)}</strong>
                            <span
                              className={"pill " + (f.drifted ? "warning" : "")}
                            >
                              {f.drifted ? "Drift detected" : "Stable"}
                            </span>
                            <p>
                              PSI <b>{f.psi.toFixed(3)}</b>
                            </p>
                            <p>
                              Mean{" "}
                              <b>
                                {f.reference_mean.toFixed(2)} →{" "}
                                {f.current_mean.toFixed(2)}
                              </b>
                            </p>
                            <p>
                              KS p-value <b>{f.p_value.toExponential(2)}</b>
                            </p>
                          </article>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="empty">
                      Generate reference and shifted datasets, then run a drift
                      report.
                    </div>
                  )}
                </>
              )}
              {view === "inference" && (
                <div className="inference-grid">
                  <section className="panel">
                    <div className="panel-heading">
                      <div>
                        <h2>A single reading. Full lineage.</h2>
                        <p>Sensor schema v1 · all features are required</p>
                      </div>
                      <Activity size={20} />
                    </div>
                    <div className="sensor-inputs">
                      {Object.entries(reading).map(([key, value]) => (
                        <label key={key}>
                          <span>{pretty(key)}</span>
                          <input
                            type="number"
                            value={value}
                            onChange={(e) =>
                              setReading((r) => ({
                                ...r,
                                [key]: Number(e.target.value),
                              }))
                            }
                            step={
                              key === "vibration" || key === "pressure"
                                ? 0.1
                                : 1
                            }
                          />
                        </label>
                      ))}
                    </div>
                    <button
                      className="primary predict-button"
                      onClick={predict}
                      disabled={busy || !champion}
                    >
                      <Play size={14} /> Score reading
                    </button>
                  </section>
                  <section className="panel prediction-panel">
                    <div className="panel-heading">
                      <h2>Prediction result</h2>
                      <span className="pill">
                        {champion
                          ? "Champion v" + champion.registry_version
                          : "No champion"}
                      </span>
                    </div>
                    {prediction ? (
                      <>
                        <div className="risk-score">
                          {(prediction.risk * 100).toFixed(1)}
                          <span>%</span>
                        </div>
                        <p>Estimated failure probability</p>
                        <span
                          className={
                            "risk-status " +
                            (prediction.failure_predicted ? "high" : "")
                          }
                        >
                          {prediction.failure_predicted
                            ? "Maintenance review recommended"
                            : "Below the operating threshold"}
                        </span>
                        <dl>
                          <div>
                            <dt>Operating threshold</dt>
                            <dd>{(prediction.threshold * 100).toFixed(1)}%</dd>
                          </div>
                          <div>
                            <dt>Registry version</dt>
                            <dd>v{prediction.registry_version}</dd>
                          </div>
                          <div>
                            <dt>Compute latency</dt>
                            <dd>{prediction.latency_ms.toFixed(2)} ms</dd>
                          </div>
                          <div>
                            <dt>Prediction ID</dt>
                            <dd>{prediction.prediction_id.slice(0, 12)}…</dd>
                          </div>
                        </dl>
                        <div className="label-actions">
                          <small>Record a delayed ground-truth label</small>
                          <button
                            disabled={busy}
                            onClick={() =>
                              action("/labels", {
                                prediction_id: prediction.prediction_id,
                                failure: 0,
                              })
                            }
                          >
                            No failure observed
                          </button>
                          <button
                            disabled={busy}
                            onClick={() =>
                              action("/labels", {
                                prediction_id: prediction.prediction_id,
                                failure: 1,
                              })
                            }
                          >
                            Failure observed
                          </button>
                        </div>
                      </>
                    ) : (
                      <div className="empty">
                        Submit a reading to inspect risk, model version, and
                        latency. This simulation is not a safety-critical
                        maintenance system.
                      </div>
                    )}
                  </section>
                </div>
              )}
            </>
          )}
          <footer>
            <span>ForgeOps · Reproducible by design</span>
            <span>
              Measured results. Explicit approvals. Traceable releases.
            </span>
          </footer>
        </div>
      </main>
      {showPipeline && (
        <div className="modal-backdrop">
          <section className="modal">
            <div className="panel-heading">
              <h2>Start a training pipeline</h2>
              <button
                aria-label="Close pipeline"
                onClick={() => setShowPipeline(false)}
              >
                <X size={18} />
              </button>
            </div>
            <p>
              Approve a dataset, train three model families, and register the
              candidates. Promotion is a separate protected action.
            </p>
            <label>
              Dataset
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
              >
                <option value="">Select the reference dataset</option>
                {data?.datasets.map((d) => (
                  <option value={d.id} key={d.id}>
                    {d.provenance.shifted ? "Shifted" : "Reference"} · {d.rows}{" "}
                    rows · {d.id.slice(0, 8)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Administrator token
              <input
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                autoComplete="off"
              />
            </label>
            <div className="modal-actions">
              <button
                className="secondary"
                disabled={busy}
                onClick={() =>
                  action("/datasets/prepare", {
                    rows: 4000,
                    seed: 42,
                    shifted: false,
                  })
                }
              >
                Prepare reference
              </button>
              <button
                className="secondary"
                disabled={busy}
                onClick={() =>
                  action("/datasets/prepare", {
                    rows: 1200,
                    seed: 77,
                    shifted: true,
                  })
                }
              >
                Prepare shifted batch
              </button>
            </div>
            <button
              className="primary"
              disabled={busy || !token}
              onClick={training}
            >
              <Play size={14} /> Approve dataset & train
            </button>
            <small>The model will not be deployed automatically.</small>
          </section>
        </div>
      )}
    </div>
  );
}
function Metric({
  label,
  value,
  note,
  icon,
  accent = false,
}: {
  label: string;
  value: string;
  note: string;
  icon: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <article className={"metric " + (accent ? "accent" : "")}>
      <div>
        <span>{label}</span>
        {icon}
      </div>
      <strong>{value}</strong>
      <p>{note}</p>
    </article>
  );
}
function Registry({
  data,
  action,
  busy,
  compact = false,
}: {
  data: Overview;
  action: (path: string, body?: unknown) => Promise<unknown>;
  busy: boolean;
  compact?: boolean;
}) {
  return (
    <section className="panel registry">
      <div className="panel-heading">
        <div>
          <h2>
            {compact ? "Registered models" : "Model comparison & promotion"}
          </h2>
          <p>Test-set evaluation · threshold selected on validation data</p>
        </div>
        <span className="subtle">{data.models.length} versions</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>MODEL</th>
              <th>VERSION</th>
              <th>PR-AUC</th>
              <th>RECALL</th>
              <th>PRECISION</th>
              <th>BRIER</th>
              <th>QUALITY GATE</th>
              <th>RELEASE</th>
            </tr>
          </thead>
          <tbody>
            {data.models.slice(0, compact ? 3 : 30).map((m) => (
              <tr key={m.id}>
                <td>
                  <span className="model-icon">
                    <Layers size={15} />
                  </span>
                  <strong>{pretty(m.kind)}</strong>
                  <small>
                    {m.id.slice(0, 8)} · {m.dataset.slice(0, 8)}
                  </small>
                </td>
                <td>v{m.registry_version}</td>
                <td>{m.metrics.pr_auc.toFixed(3)}</td>
                <td>{m.metrics.recall.toFixed(3)}</td>
                <td>{m.metrics.precision.toFixed(3)}</td>
                <td>{m.metrics.brier.toFixed(3)}</td>
                <td>
                  <span
                    title={m.gate.reasons.join("; ")}
                    className={"pill " + (!m.gate.passed ? "warning" : "")}
                  >
                    <span />
                    {m.gate.passed ? "Passed minimums" : "Rejected"}
                  </span>
                </td>
                <td>
                  {m.id === data.champion ? (
                    <span className="champion">
                      <ShieldCheck size={12} /> Champion
                    </span>
                  ) : !compact && m.gate.passed ? (
                    <button
                      className="promote"
                      disabled={busy}
                      onClick={() =>
                        action(data.champion ? "/rollouts/start" : "/models/promote", { model_id: m.id })
                      }
                    >
                      {data.champion ? "Approve 20% canary" : "Approve initial champion"} <ArrowUpRight size={12} />
                    </button>
                  ) : (
                    <span className="subtle">Candidate</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!data.models.length && (
          <div className="empty">
            No models registered. Prepare a dataset and run the pipeline.
          </div>
        )}
      </div>
    </section>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
