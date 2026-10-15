import { useId } from 'react';
import { LuCircleAlert } from 'react-icons/lu';

// Label + control + hint + error, wired together for screen readers.
export default function FormField({ label, name, type = 'text', as = 'input', options, value, onChange, error, hint, prefix, className = '', ...rest }) {
  const id = useId();
  const describedBy = [hint && `${id}-hint`, error && `${id}-err`].filter(Boolean).join(' ') || undefined;
  const common = {
    id, name, value, onChange, 'aria-invalid': error ? 'true' : undefined, 'aria-describedby': describedBy, ...rest,
  };
  let control;
  if (as === 'select') {
    control = (
      <select className="select" {...common}>
        <option value="" disabled>Select…</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    );
  } else if (as === 'textarea') {
    control = <textarea className="textarea" {...common} />;
  } else {
    control = (
      <div className="input-wrap">
        {prefix && <span className="prefix" aria-hidden="true">{prefix}</span>}
        <input className={`input${prefix ? ' has-prefix' : ''}`} type={type} {...common} />
      </div>
    );
  }
  return (
    <div className={`field${error ? ' invalid' : ''} ${className}`}>
      <label htmlFor={id}>{label}</label>
      {control}
      {hint && <p className="hint" id={`${id}-hint`}>{hint}</p>}
      {error && (
        <p className="field-error" id={`${id}-err`} role="alert">
          <LuCircleAlert size={16} aria-hidden="true" style={{ flex: 'none', marginTop: 2 }} />
          {error}
        </p>
      )}
    </div>
  );
}
