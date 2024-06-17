import logo from './logo.svg';
import './index';
import { BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import Home from './pages/Home';
import PageMenu from './pages/PageMenu';
import Service from './pages/Service';
import ResultPage from './pages/ResultPage';
import AddressPage from './pages/AddressPage';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/address" element={<AddressPage />} />
        <Route path="/menu" element={<PageMenu />} />
        <Route path="/menu/taskassignments" element={<Service />} />
        <Route path="/menu/result" element={<ResultPage />} />
      </Routes>
    </Router>
  );
}

export default App;
