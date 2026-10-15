import { LuShieldCheck } from 'react-icons/lu';
import { Link } from '../router.jsx';

// Only technologies that actually appear in the project's code are listed.
const STACK = ['Python', 'pandas', 'NumPy', 'scikit-learn', 'networkx', 'joblib', 'Flask', 'React', 'Vite'];

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div>
            <span className="logo">
              <span className="logo-mark"><LuShieldCheck size={20} aria-hidden="true" /></span>
              <span>FraudShield <em>AI</em></span>
            </span>
            <p style={{ marginTop: '1rem' }}>
              A machine-learning project that scores insurance claims for fraud risk and
              helps prioritize which ones a person should review.
            </p>
          </div>
          <nav aria-label="Footer">
            <h3>Explore</h3>
            <ul>
              <li><Link to="/">Home</Link></li>
              <li><Link to="/about">About the project</Link></li>
              <li><Link to="/how-it-works">How it works</Link></li>
              <li><Link to="/insights">Model insights</Link></li>
              <li><Link to="/analyze">Analyze a claim</Link></li>
            </ul>
          </nav>
          <div>
            <h3>Built with</h3>
            <div className="chip-row">
              {STACK.map((s) => <span className="chip" key={s}>{s}</span>)}
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p>
            FraudShield AI is an academic project. It was trained and tested on a synthetic
            dataset, is not a production system, and is not connected to any insurer. Model
            outputs support a human reviewer and are not proof of fraud.
          </p>
        </div>
      </div>
    </footer>
  );
}
