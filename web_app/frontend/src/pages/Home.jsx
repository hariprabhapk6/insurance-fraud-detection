import { LuArrowRight, LuGauge, LuBrain, LuScanSearch, LuUserCheck } from 'react-icons/lu';
import Button from '../components/Button.jsx';
import HeroVisual from '../components/HeroVisual.jsx';
import FeatureCard from '../components/FeatureCard.jsx';
import SectionHeading from '../components/SectionHeading.jsx';
import { Link } from '../router.jsx';
import metrics from '../data/metrics.json';

const rf = metrics.ml.random_forest.metrics;

export default function Home() {
  return (
    <>
      <section className="hero">
        <div className="container hero-grid">
          <div className="hero-copy">
            <h1>Detect Insurance Fraud Before It Costs You.</h1>
            <p className="hero-lead">
              An AI-powered insurance claims analysis system that evaluates claim information,
              identifies suspicious patterns, and supports faster, data-driven fraud investigations.
            </p>
            <div className="hero-actions">
              <Button to="/analyze">Analyze a Claim <LuArrowRight size={18} aria-hidden="true" /></Button>
              <Button to="/how-it-works" variant="secondary">Explore How It Works</Button>
            </div>
            <p className="hero-note">
              Academic project trained on a synthetic dataset of {metrics.dataset.total_claims.toLocaleString()} claims.
            </p>
          </div>
          <HeroVisual />
        </div>
      </section>

      <section className="strip" aria-label="Project at a glance">
        <div className="container strip-grid">
          <p><strong>{metrics.dataset.total_claims.toLocaleString()}</strong> claims in the dataset, <strong>{(metrics.dataset.fraud_rate * 100).toFixed(1)}%</strong> labelled fraud</p>
          <p><strong>{rf.roc_auc.toFixed(3)}</strong> ROC-AUC for the Random Forest on {metrics.dataset.test_size.toLocaleString()} held-out claims</p>
          <p><strong>16</strong> inputs per claim: 11 entered, 5 derived from them</p>
        </div>
      </section>

      <section className="section" aria-labelledby="features-title">
        <div className="container">
          <SectionHeading id="features-title" title="What the project can do">
            Each capability below is backed by code in the project. The label shows how it is used on this site.
          </SectionHeading>
          <div className="feature-grid">
            <FeatureCard icon={LuGauge} title="Claim risk analysis" status="live">
              Enter a claim&rsquo;s details and receive a fraud score, a predicted class, and a plain-language reading of what it means.
            </FeatureCard>
            <FeatureCard icon={LuBrain} title="Machine learning detection" status="live">
              A Random Forest of 300 trees, trained with class weights to cope with how rare fraud is, and tested on claims it never saw.
            </FeatureCard>
            <FeatureCard icon={LuScanSearch} title="Suspicious pattern identification" status="experimental">
              Rule-based indicators (high-value amount, brand-new policy, repeat claimant) and a text classifier that reads the claim description. The text model learned from templated, synthetic wording.
            </FeatureCard>
            <FeatureCard icon={LuUserCheck} title="AI-assisted investigation" status="research">
              An agent prototype combines the models and sends uncertain or high-risk claims to a human reviewer. It runs in the project code, not on this site.
            </FeatureCard>
          </div>
        </div>
      </section>

      <section className="cta-band">
        <div className="container cta-inner">
          <div>
            <h2>See how a claim gets scored</h2>
            <p>Load a sample claim from the dataset or enter your own, and read the result.</p>
          </div>
          <div className="cta-actions">
            <Button to="/analyze" variant="on-dark">Analyze a Claim</Button>
            <Link to="/insights" className="cta-link">View model insights</Link>
          </div>
        </div>
      </section>
    </>
  );
}
