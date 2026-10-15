// status: 'live' = used by the Analyze form, 'experimental' = works but limited, 'research' = in the project code only
const STATUS = {
  live: ['tag-live', 'Used in the demo'],
  experimental: ['tag-exp', 'Experimental'],
  research: ['tag-research', 'Research prototype'],
};

export default function FeatureCard({ icon: Icon, title, status, children }) {
  const [cls, label] = STATUS[status];
  return (
    <article className="feature-card">
      <div className="feature-icon"><Icon size={22} aria-hidden="true" /></div>
      <h3>{title}</h3>
      <p>{children}</p>
      <span className={`tag ${cls}`}>{label}</span>
    </article>
  );
}
