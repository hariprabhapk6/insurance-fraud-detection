import { useEffect, useState } from 'react';
import { LuShieldCheck, LuMenu, LuX } from 'react-icons/lu';
import { Link } from '../router.jsx';
import Button from './Button.jsx';

const LINKS = [
  { to: '/', label: 'Home' },
  { to: '/about', label: 'About' },
  { to: '/how-it-works', label: 'How It Works' },
  { to: '/insights', label: 'Model Insights' },
];

export default function Navbar({ route }) {
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [route]);
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && setOpen(false);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <header className="nav">
      <div className="container nav-inner">
        <Link to="/" className="logo" aria-label="FraudShield AI, home">
          <span className="logo-mark"><LuShieldCheck size={20} aria-hidden="true" /></span>
          <span>FraudShield <em>AI</em></span>
        </Link>
        <button
          className="nav-toggle"
          aria-expanded={open}
          aria-controls="main-nav"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen((o) => !o)}
        >
          {open ? <LuX size={22} /> : <LuMenu size={22} />}
        </button>
        <nav aria-label="Main">
          <ul id="main-nav" className={`nav-links${open ? ' open' : ''}`}>
            {LINKS.map((l) => (
              <li key={l.to}>
                <Link to={l.to} aria-current={route === l.to ? 'page' : undefined}>{l.label}</Link>
              </li>
            ))}
            <li>
              <Button to="/analyze" size="sm">Analyze a Claim</Button>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}
