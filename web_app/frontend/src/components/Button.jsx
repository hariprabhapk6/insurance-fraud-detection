import { Link } from '../router.jsx';

// One button component so every action on the site looks and behaves the same.
export default function Button({ to, variant = 'primary', size, className = '', children, ...rest }) {
  const cls = `btn btn-${variant}${size ? ` btn-${size}` : ''} ${className}`.trim();
  if (to) {
    return (
      <Link to={to} className={cls} {...rest}>
        {children}
      </Link>
    );
  }
  return (
    <button type="button" className={cls} {...rest}>
      {children}
    </button>
  );
}
