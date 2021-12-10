
import './App.css';
import Navbar from './components/Navbar';
import HomePage from './components/HomePage';
import Footer from './components/Footer';
import {BrowserRouter as Router, Route} from 'react-router-dom'



function App() {
  return (
    <div className="App">
      <Navbar/>
      <HomePage/>
      <Footer/>
    </div>
  );
}   

export default App;
