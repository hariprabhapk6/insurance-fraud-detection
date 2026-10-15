// Horizontal 0-100% meter with the model's decision line and the UI "medium" band marked.
export default function ScoreMeter({ value, mediumFrom, highFrom, level }) {
  const pct = Math.round(value * 1000) / 10;
  return (
    <div className="meter" role="img" aria-label={`Fraud score ${pct} percent. Review starts at ${highFrom * 100} percent.`}>
      <div className="meter-track">
        <span className="meter-band meter-low" style={{ width: `${mediumFrom * 100}%` }} />
        <span className="meter-band meter-med" style={{ left: `${mediumFrom * 100}%`, width: `${(highFrom - mediumFrom) * 100}%` }} />
        <span className="meter-band meter-high" style={{ left: `${highFrom * 100}%`, right: 0 }} />
        <span className={`meter-marker marker-${level}`} style={{ left: `${Math.min(Math.max(value, 0), 1) * 100}%` }} />
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span style={{ left: 0 }}>0%</span>
        <span style={{ left: `${highFrom * 100}%` }}>{highFrom * 100}% model threshold</span>
        <span style={{ right: 0 }}>100%</span>
      </div>
    </div>
  );
}
