// Small, dependency-free chart components built from plain HTML/SVG.

export function StatTile({ label, value, note }) {
  return (
    <div className="stat-tile">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}

// Two-class split shown as one bar, with a text equivalent.
export function ClassSplit({ fraud, notFraud }) {
  const total = fraud + notFraud;
  const fraudPct = (fraud / total) * 100;
  return (
    <div>
      <div className="split-bar" role="img" aria-label={`${notFraud} not fraud and ${fraud} fraud out of ${total} claims`}>
        <span className="split-ok" style={{ width: `${100 - fraudPct}%` }} />
        <span className="split-fraud" style={{ width: `${fraudPct}%` }} />
      </div>
      <ul className="legend">
        <li><i className="dot dot-ok" /> Not fraud: <strong>{notFraud.toLocaleString()}</strong> ({(100 - fraudPct).toFixed(1)}%)</li>
        <li><i className="dot dot-fraud" /> Fraud: <strong>{fraud.toLocaleString()}</strong> ({fraudPct.toFixed(1)}%)</li>
      </ul>
    </div>
  );
}

export function ConfusionMatrix({ tn, fp, fn, tp }) {
  const rows = [
    { label: 'Actually not fraud', cells: [{ v: tn, k: 'ok', t: 'Correctly cleared' }, { v: fp, k: 'bad', t: 'False alarm' }] },
    { label: 'Actually fraud', cells: [{ v: fn, k: 'bad', t: 'Missed fraud' }, { v: tp, k: 'ok', t: 'Caught fraud' }] },
  ];
  return (
    <div className="cm" role="table" aria-label="Confusion matrix on the 1,200-claim test set">
      <div className="cm-corner" />
      <div className="cm-col" role="columnheader">Predicted not fraud</div>
      <div className="cm-col" role="columnheader">Predicted fraud</div>
      {rows.map((r) => (
        <div className="cm-row" role="row" key={r.label}>
          <div className="cm-rowlabel" role="rowheader">{r.label}</div>
          {r.cells.map((c) => (
            <div key={c.t} role="cell" className={`cm-cell cm-${c.k}`}>
              <strong>{c.v.toLocaleString()}</strong>
              <span>{c.t}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// Horizontal bars; each item is {label, value, display?, tone?}
export function BarList({ items, max, unit = '' }) {
  const top = max ?? Math.max(...items.map((i) => i.value));
  return (
    <ul className="barlist">
      {items.map((i) => (
        <li key={i.label}>
          <span className="bar-label">{i.label}</span>
          <span className="bar-track">
            <span className={`bar-fill ${i.tone ? `bar-${i.tone}` : ''}`} style={{ width: `${(i.value / top) * 100}%` }} />
          </span>
          <span className="bar-value">{i.display ?? i.value.toFixed(3)}{unit}</span>
        </li>
      ))}
    </ul>
  );
}
