import { LuTriangleAlert, LuCircleCheck, LuFileSearch, LuFlaskConical, LuInfo, LuLoaderCircle, LuScanSearch } from 'react-icons/lu';
import ScoreMeter from './ScoreMeter.jsx';

const LEVELS = {
  high: {
    title: 'Potentially suspicious',
    Icon: LuTriangleAlert,
    meaning: 'The score is at or above the model’s 50% decision threshold, so this claim is flagged. Treat it as a reason for a person to look closer, not as a finding of fraud.',
  },
  medium: {
    title: 'Some risk signals',
    Icon: LuFileSearch,
    meaning: 'The score is below the model’s decision threshold, so the claim is not flagged, but it sits in the upper part of the range. Routine checks are enough unless other information raises concern.',
  },
  low: {
    title: 'Low risk',
    Icon: LuCircleCheck,
    meaning: 'The score is well below the model’s decision threshold and the claim is not flagged. This does not guarantee the claim is genuine.',
  },
};

const pct = (n) => `${(n * 100).toFixed(1)}%`;

export default function ResultPanel({ status, result }) {
  if (status === 'loading') {
    return (
      <div className="result result-empty" aria-live="polite">
        <LuLoaderCircle size={34} className="spin" aria-hidden="true" />
        <h3>Analyzing the claim…</h3>
        <p>The Random Forest model is scoring your inputs.</p>
      </div>
    );
  }
  if (status !== 'done' || !result) {
    return (
      <div className="result result-empty">
        <LuScanSearch size={34} aria-hidden="true" />
        <h3>Your result will appear here</h3>
        <p>
          Fill in the claim details, or load a sample claim from the dataset, then select
          Analyze Claim.
        </p>
      </div>
    );
  }

  const level = LEVELS[result.risk_level];
  const { Icon } = level;
  return (
    <div className={`result result-${result.risk_level}`} aria-live="polite">
      <div className="result-head">
        <span className="result-icon"><Icon size={26} aria-hidden="true" /></span>
        <div>
          <h3>{level.title}</h3>
          <p className="result-class">
            Predicted class: <strong>{result.predicted_class === 1 ? '1 (fraud)' : '0 (not fraud)'}</strong>
          </p>
        </div>
      </div>

      <div className="score-row">
        <span className="score-value">{pct(result.fraud_probability)}</span>
        <span className="score-label">model fraud score</span>
      </div>
      <ScoreMeter
        value={result.fraud_probability}
        mediumFrom={result.risk_bands.medium_from}
        highFrom={result.risk_bands.high_from}
        level={result.risk_level}
      />

      <p className="result-meaning">{level.meaning}</p>

      {result.warnings.length > 0 && (
        <div className="notice notice-warn">
          <LuTriangleAlert size={18} aria-hidden="true" />
          <div>{result.warnings.map((w) => <p key={w}>{w}</p>)}</div>
        </div>
      )}

      {result.indicators.length > 0 && (
        <section className="result-block">
          <h4>Rule-based indicators</h4>
          <ul>{result.indicators.map((i) => <li key={i}>{i}</li>)}</ul>
        </section>
      )}

      <section className="result-block">
        <h4>
          Description text signal <span className="tag tag-exp">Experimental</span>
        </h4>
        <p>
          A separate text classifier gives the description a score of{' '}
          <strong>{pct(result.text_signal.probability)}</strong>
          {result.text_signal.keywords.length > 0 && (
            <> and picks out: {result.text_signal.keywords.map((k) => <code key={k}>{k}</code>)}</>
          )}
          .
        </p>
        <p className="result-fine">
          <LuFlaskConical size={14} aria-hidden="true" /> It was trained on a small set of templated
          descriptions, so it is unreliable for free-form text. It does not change the score above.
        </p>
      </section>

      <details className="result-block">
        <summary>Features the model derived from your inputs</summary>
        <dl className="derived">
          <div><dt>Claim-to-income ratio</dt><dd>{result.derived_features.claim_to_income_ratio.toFixed(3)}</dd></div>
          <div><dt>Policy tenure bucket</dt><dd>{result.derived_features.tenure_bucket}</dd></div>
          <div><dt>Day of week (0 = Monday)</dt><dd>{result.derived_features.claim_dow}</dd></div>
          <div><dt>Month</dt><dd>{result.derived_features.claim_month}</dd></div>
          <div><dt>High-value claim</dt><dd>{result.derived_features.is_high_value_claim ? 'Yes' : 'No'}</dd></div>
          <div><dt>Description length</dt><dd>{result.derived_features.desc_length} characters, {result.derived_features.desc_word_count} words</dd></div>
        </dl>
      </details>

      <div className="notice">
        <LuInfo size={18} aria-hidden="true" />
        <p>
          <strong>Decision support only.</strong> This score comes from {result.model}. It ranks
          claims by suspicion but is not a calibrated probability, and it is not proof of
          fraud. A trained investigator should make any decision about a claim.
        </p>
      </div>
    </div>
  );
}
