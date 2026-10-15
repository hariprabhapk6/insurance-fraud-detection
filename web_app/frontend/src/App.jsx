import { useEffect, useRef } from 'react';
import { useRoute } from './router.jsx';
import Navbar from './components/Navbar.jsx';
import Footer from './components/Footer.jsx';
import Home from './pages/Home.jsx';
import About from './pages/About.jsx';
import HowItWorks from './pages/HowItWorks.jsx';
import Analyze from './pages/Analyze.jsx';
import Insights from './pages/Insights.jsx';

const PAGES = {
  '/': { C: Home, title: 'FraudShield AI | Insurance claims fraud detection' },
  '/about': { C: About, title: 'About the project | FraudShield AI' },
  '/how-it-works': { C: HowItWorks, title: 'How it works | FraudShield AI' },
  '/analyze': { C: Analyze, title: 'Analyze a claim | FraudShield AI' },
  '/insights': { C: Insights, title: 'Model insights | FraudShield AI' },
};

export default function App() {
  const route = useRoute();
  const page = PAGES[route] || PAGES['/'];
  const main = useRef(null);

  // On navigation: update the tab title, scroll to top, and move focus to the content
  // so keyboard and screen-reader users start at the new page.
  useEffect(() => {
    document.title = page.title;
    window.scrollTo(0, 0);
    main.current?.focus({ preventScroll: true });
  }, [route, page.title]);

  const { C } = page;
  return (
    <>
      <a className="skip-link" href="#main" onClick={(e) => { e.preventDefault(); main.current?.focus(); }}>Skip to content</a>
      <Navbar route={route in PAGES ? route : '/'} />
      <main id="main" tabIndex={-1} ref={main} style={{ outline: 'none' }}>
        <C />
      </main>
      <Footer />
    </>
  );
}
