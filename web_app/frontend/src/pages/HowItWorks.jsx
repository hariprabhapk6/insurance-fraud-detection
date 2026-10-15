import { LuClipboardList, LuDatabase, LuCpu, LuGauge, LuUserCheck } from 'react-icons/lu';
import Button from '../components/Button.jsx';

const STEPS = [
  { icon: LuClipboardList, title: 'Claim information', text: 'You provide the claim: type, date, amount and description, plus the policyholder’s age, income, gender, occupation and city, and the policy’s age and prior claims.' },
  { icon: LuDatabase, title: 'Data preparation', text: 'The service derives the features the model was trained on: claim-to-income ratio, tenure bucket, day and month, a high-value flag, and description length. The model’s pipeline then scales numbers and one-hot encodes categories.' },
  { icon: LuCpu, title: 'AI analysis', text: 'A Random Forest of 300 decision trees scores the claim. A separate TF-IDF text classifier also reads the description, as an experimental extra signal.' },
  { icon: LuGauge, title: 'Risk assessment', text: 'You see the model’s fraud score, the predicted class (flagged at 50% or above), and a risk band. The band cut-offs are a display choice for this site.' },
  { icon: LuUserCheck, title: 'Investigation support', text: 'Flagged claims are presented as potentially suspicious for a person to review, with rule-based indicators such as a very new policy. The site never decides to deny a claim.' },
];

export default function HowItWorks() {
  return (
    <>
      <section className="page-head">
        <div className="container">
          <h1>From claim details to a risk score in five steps</h1>
          <p>What happens between selecting Analyze Claim and seeing a result, mapped to the project&rsquo;s real code.</p>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <ol className="stepper">
            {STEPS.map(({ icon: Icon, title, text }, i) => (
              <li key={title}>
                <span className="step-node"><Icon size={22} aria-hidden="true" /><b>{i + 1}</b></span>
                <h3>{title}</h3>
                <p>{text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="section section-grey">
        <div className="container two-col">
          <div>
            <h2>What is not in this flow</h2>
            <p>
              The project also contains an event-sequence RNN, a policyholder-network GCN and an investigation agent.
              They need data a single form cannot provide (a claim&rsquo;s event history, links between policyholders), so they are
              not part of the live analysis. Their results appear on the Model Insights page.
            </p>
          </div>
          <div>
            <h2>Try it yourself</h2>
            <p>Load a sample claim from the dataset and watch the steps above produce a result.</p>
            <Button to="/analyze">Analyze a Claim</Button>
          </div>
        </div>
      </section>
    </>
  );
}
