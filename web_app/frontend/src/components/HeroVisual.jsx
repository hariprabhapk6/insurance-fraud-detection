// Original illustration: a claim file being read by the model.
// It is a drawing, not live output, and is captioned as an illustration.
export default function HeroVisual() {
  return (
    <figure className="hero-visual" aria-label="Illustration of a claim file with risky phrases highlighted and a risk gauge">
      <div className="claim-sheet">
        <div className="scanline" aria-hidden="true" />
        <div className="sheet-top">
          <span className="sheet-title">Claim file</span>
          <span className="sheet-pill">Under review</span>
        </div>
        <dl className="sheet-fields">
          <div><dt>Type</dt><dd>Property</dd></div>
          <div><dt>Policy age</dt><dd className="hl hl-amber">1 month</dd></div>
          <div><dt>Prior claims</dt><dd>0</dd></div>
        </dl>
        <p className="sheet-text">
          Property damage claim filed <mark className="hl hl-red">right before policy renewal</mark>,
          inflated repair estimate from an <mark className="hl hl-red">unlisted vendor</mark>.
        </p>
        <div className="sheet-lines" aria-hidden="true"><i /><i /><i /></div>
      </div>
      <div className="gauge-card">
        <svg viewBox="0 0 160 96" role="img" aria-label="Risk gauge pointing to high">
          <path d="M16 84 A64 64 0 0 1 56 25" fill="none" stroke="#2ea36f" strokeWidth="12" strokeLinecap="round" />
          <path d="M64 21 A64 64 0 0 1 100 22" fill="none" stroke="#e0a21b" strokeWidth="12" strokeLinecap="round" />
          <path d="M108 26 A64 64 0 0 1 144 84" fill="none" stroke="#d64b4b" strokeWidth="12" strokeLinecap="round" />
          <g className="needle"><line x1="80" y1="84" x2="80" y2="36" stroke="#0b1f3a" strokeWidth="4" strokeLinecap="round" /></g>
          <circle cx="80" cy="84" r="8" fill="#0b1f3a" />
        </svg>
        <span className="gauge-label">Flagged for review</span>
      </div>
      <figcaption>Illustration only. Real scores come from the Analyze page.</figcaption>
    </figure>
  );
}
