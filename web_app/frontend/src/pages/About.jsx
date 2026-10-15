
import { LuTarget, LuSearchCheck, LuLayers } from 'react-icons/lu';
import SectionHeading from '../components/SectionHeading.jsx';

export default function About() {
  return (
    <>
      <section className="page-head">
        <div className="container">
          <h1>About FraudShield</h1>
          <p>Smarter Claim Analysis. Safer Decisions.</p>
        </div>
      </section>

      <section className="section">
        <div className="container prose-grid">
          <div>
            <h2>What is FraudShield?</h2>
            <p>
              FraudShield is an AI-powered insurance fraud detection
              system that analyzes claims, highlights suspicious
              patterns, and helps investigators prioritize reviews.
            </p>
            <p>
              Our goal is to make claim investigation smarter and
              more efficient through Machine Learning.
            </p>
          </div>

          <aside className="card callout">
            <LuTarget size={22} aria-hidden="true" />
            <h3>Our Objectives</h3>
            <ul>
              <li>Analyze insurance claim details.</li>
              <li>Identify potentially suspicious patterns.</li>
              <li>Generate risk scores to support claim review.</li>
              <li>Keep final decisions under human control.</li>
            </ul>
          </aside>
        </div>
      </section>

      <section className="section section-grey">
        <div className="container two-col">
          <div>
            <h2>
              <LuSearchCheck
                size={24}
                aria-hidden="true"
                style={{ verticalAlign: '-4px', marginRight: 8 }}
              />
              How It Works
            </h2>
            <p>
              The Random Forest Classifier analyzes claim details
              and estimates fraud risk. The project also explores
              NLP-based text analysis and other experimental
              approaches.
            </p>
          </div>

          <div>
            <h2>
              <LuLayers
                size={24}
                aria-hidden="true"
                style={{ verticalAlign: '-4px', marginRight: 8 }}
              />
              Built With
            </h2>
            <div className="tech-tags">
              <span>Python</span>
              <span>Machine Learning</span>
              <span>NLP</span>
              <span>Flask</span>
              <span>React</span>
              <span>Vite</span>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="card">
            <h3>Our Goal</h3>
            <p>
              To support faster, smarter, and more responsible
              insurance claim investigation through AI.
            </p>
            <p>
              <strong>
                #FraudShield #MachineLearning #FraudDetection
              </strong>
            </p>
            <p>
              <small>
                A risk score supports investigation; it does not
                confirm fraud.
              </small>
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
