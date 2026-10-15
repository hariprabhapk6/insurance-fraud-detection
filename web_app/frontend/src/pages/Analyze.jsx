import { useEffect, useRef, useState } from 'react';
import { LuLoaderCircle, LuRotateCcw, LuCircleAlert, LuSparkles, LuArrowRight } from 'react-icons/lu';
import Button from '../components/Button.jsx';
import FormField from '../components/FormField.jsx';
import ResultPanel from '../components/ResultPanel.jsx';
import { getSchema, predictClaim, ApiError } from '../services/api.js';
import samples from '../data/samples.json';

const EMPTY = {
  claim_type: '', claim_date: '', claim_amount: '', claim_description: '',
  claimant_age: '', gender: '', occupation: '', city: '', annual_income: '',
  policy_tenure_months: '', prior_claims_count: '',
};

// Used only if the backend cannot be reached when the page loads; they mirror the
// categories the model was trained on, so the form still renders for a UI preview.
const FALLBACK = {
  claim_type: ['Auto', 'Health', 'Life', 'Property', 'Travel'],
  gender: ['F', 'M', 'Other', 'Unknown'],
  occupation: ['Business Owner', 'Retired', 'Salaried', 'Self-Employed', 'Student', 'Unemployed', 'Unknown'],
  city: ['Ahmedabad', 'Bengaluru', 'Chennai', 'Coimbatore', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata', 'Mumbai', 'Pune', 'Unknown'],
};
const GENDER_LABEL = { F: 'Female', M: 'Male', Other: 'Other', Unknown: 'Prefer not to say' };
const opt = (key, v) => ({ value: v, label: key === 'gender' ? GENDER_LABEL[v] || v : v === 'Unknown' ? 'Unknown / not provided' : v });
const SAMPLE_NAMES = ['Auto collision', 'Hospital surgery', 'Stolen vehicle', 'Property damage'];

// Client-side checks mirror the server's rules so most mistakes are caught instantly.
function validate(f) {
  const e = {};
  const need = (k, msg) => { if (String(f[k]).trim() === '') e[k] = msg; };
  const range = (k, label, lo, hi, int) => {
    need(k, `${label} is required.`);
    if (e[k]) return;
    const n = Number(f[k]);
    if (!Number.isFinite(n)) e[k] = `${label} must be a number.`;
    else if (int && !Number.isInteger(n)) e[k] = `${label} must be a whole number.`;
    else if (n < lo || n > hi) e[k] = `${label} must be between ${lo.toLocaleString()} and ${hi.toLocaleString()}.`;
  };
  ['claim_type', 'gender', 'occupation', 'city'].forEach((k) => need(k, 'Choose an option.'));
  range('claimant_age', 'Age', 18, 100, true);
  range('annual_income', 'Annual income', 1, 100000000);
  range('policy_tenure_months', 'Policy tenure', 0, 600, true);
  range('claim_amount', 'Claim amount', 1, 100000000);
  range('prior_claims_count', 'Prior claims', 0, 50, true);
  if (!f.claim_date) e.claim_date = 'Enter the claim date.';
  else if (f.claim_date > new Date().toISOString().slice(0, 10)) e.claim_date = 'Claim date cannot be in the future.';
  const len = f.claim_description.trim().length;
  if (len < 10) e.claim_description = 'Describe the claim in at least 10 characters.';
  else if (len > 500) e.claim_description = 'Keep the description under 500 characters.';
  return e;
}

export default function Analyze() {
  const [form, setForm] = useState(EMPTY);
  const [errors, setErrors] = useState({});
  const [status, setStatus] = useState('idle'); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [options, setOptions] = useState(FALLBACK);
  const [schemaDown, setSchemaDown] = useState(false);
  const busy = useRef(false); // blocks duplicate submissions even before React re-renders
  const resultRef = useRef(null);
  const today = new Date().toISOString().slice(0, 10);

  useEffect(() => {
    getSchema().then((s) => setOptions(s.options)).catch(() => setSchemaDown(true));
  }, []);

  const set = (e) => {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: value }));
    if (errors[name]) setErrors((x) => ({ ...x, [name]: undefined }));
  };

  const reset = () => { setForm(EMPTY); setErrors({}); setResult(null); setApiError(null); setStatus('idle'); };

  const loadSample = (i) => {
    const s = samples[i];
    setForm(Object.fromEntries(Object.entries(s).map(([k, v]) => [k, String(v)])));
    setErrors({}); setResult(null); setApiError(null); setStatus('idle');
  };

  const submit = async (ev) => {
    ev.preventDefault();
    if (busy.current) return;
    const found = validate(form);
    setErrors(found);
    if (Object.keys(found).length) {
      document.getElementById('claim-form')?.querySelector('[aria-invalid="true"]')?.focus();
      return;
    }
    busy.current = true;
    setStatus('loading'); setApiError(null); setResult(null);
    if (window.matchMedia('(max-width: 960px)').matches) resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      const data = await predictClaim({
        ...form,
        claimant_age: Number(form.claimant_age), annual_income: Number(form.annual_income),
        policy_tenure_months: Number(form.policy_tenure_months), claim_amount: Number(form.claim_amount),
        prior_claims_count: Number(form.prior_claims_count),
      });
      setResult(data); setStatus('done');
    } catch (err) {
      const e = err instanceof ApiError ? err : new ApiError('Something went wrong. Try again.');
      setErrors((x) => ({ ...x, ...e.fieldErrors }));
      setApiError(e); setStatus('error');
    } finally {
      busy.current = false;
    }
  };

  const loading = status === 'loading';
  const sel = (k) => options[k].map((v) => opt(k, v));

  return (
    <>
      <section className="page-head">
        <div className="container">
          <h1>Analyze a claim</h1>
          <p>Enter the details the model was trained on. It returns a fraud score for a person to weigh, not a verdict.</p>
        </div>
      </section>

      <section className="section analyze">
        <div className="container analyze-grid">
          <form id="claim-form" className="card claim-form" onSubmit={submit} noValidate aria-busy={loading}>
            <div className="sample-row">
              <span className="sample-label"><LuSparkles size={16} aria-hidden="true" /> Try a sample claim from the dataset</span>
              <div className="chip-row">
                {samples.map((s, i) => (
                  <button type="button" key={SAMPLE_NAMES[i]} className="chip chip-btn" onClick={() => loadSample(i)} disabled={loading}>
                    {SAMPLE_NAMES[i]}
                  </button>
                ))}
              </div>
            </div>

            {schemaDown && (
              <div className="notice notice-warn" role="status">
                <LuCircleAlert size={18} aria-hidden="true" />
                <p>The analysis service is not running, so scoring is unavailable. Start the backend (see the README), then reload.</p>
              </div>
            )}

            <fieldset>
              <legend>Claim details</legend>
              <div className="fields">
                <FormField as="select" label="Claim type" name="claim_type" value={form.claim_type} onChange={set} error={errors.claim_type} options={sel('claim_type')} />
                <FormField label="Claim date" name="claim_date" type="date" max={today} value={form.claim_date} onChange={set} error={errors.claim_date} />
                <FormField label="Claim amount" name="claim_amount" type="number" inputMode="decimal" min="1" step="0.01" prefix="₹" placeholder="e.g. 45000" value={form.claim_amount} onChange={set} error={errors.claim_amount} />
                <FormField as="textarea" className="span-2" label="Claim description" name="claim_description" maxLength={500} placeholder="Describe what happened, in a sentence or two." value={form.claim_description} onChange={set} error={errors.claim_description}
                  hint={`${form.claim_description.length}/500 characters. The model uses its length and word count.`} />
              </div>
            </fieldset>

            <fieldset>
              <legend>Policyholder</legend>
              <div className="fields">
                <FormField label="Age" name="claimant_age" type="number" inputMode="numeric" min="18" max="100" step="1" placeholder="e.g. 42" value={form.claimant_age} onChange={set} error={errors.claimant_age} />
                <FormField as="select" label="Gender" name="gender" value={form.gender} onChange={set} error={errors.gender} options={sel('gender')} />
                <FormField as="select" label="Occupation" name="occupation" value={form.occupation} onChange={set} error={errors.occupation} options={sel('occupation')} />
                <FormField as="select" label="City" name="city" value={form.city} onChange={set} error={errors.city} options={sel('city')} />
                <FormField label="Annual income" name="annual_income" type="number" inputMode="decimal" min="1" step="0.01" prefix="₹" placeholder="e.g. 650000" value={form.annual_income} onChange={set} error={errors.annual_income} />
              </div>
            </fieldset>

            <fieldset>
              <legend>Policy history</legend>
              <div className="fields">
                <FormField label="Policy tenure" name="policy_tenure_months" type="number" inputMode="numeric" min="0" step="1" placeholder="e.g. 24" hint="Months since the policy started." value={form.policy_tenure_months} onChange={set} error={errors.policy_tenure_months} />
                <FormField label="Prior claims" name="prior_claims_count" type="number" inputMode="numeric" min="0" step="1" placeholder="e.g. 0" hint="Claims filed before this one." value={form.prior_claims_count} onChange={set} error={errors.prior_claims_count} />
              </div>
            </fieldset>

            {status === 'error' && apiError && (
              <div className="notice notice-error" role="alert">
                <LuCircleAlert size={18} aria-hidden="true" />
                <p>
                  {apiError.network
                    ? 'Could not reach the analysis service. Start the backend with python3 web_app/backend/app.py, then try again.'
                    : apiError.message}
                </p>
              </div>
            )}

            <div className="form-actions">
              <Button type="submit" disabled={loading} aria-disabled={loading}>
                {loading ? <><LuLoaderCircle size={18} className="spin" aria-hidden="true" /> Analyzing…</> : <>Analyze Claim <LuArrowRight size={18} aria-hidden="true" /></>}
              </Button>
              <Button variant="ghost" onClick={reset} disabled={loading}><LuRotateCcw size={16} aria-hidden="true" /> Clear Form</Button>
            </div>
          </form>

          <div className="result-col" ref={resultRef}>
            <ResultPanel status={status} result={result} />
          </div>
        </div>
      </section>
    </>
  );
}
