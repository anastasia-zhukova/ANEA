
import './App.css';
import Navbar from './components/Navbar';
import HomePage from './components/HomePage';
import Footer from './components/Footer';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import AppBody from './components/AppBody';



function App() {
  return (
    <div className="App">
      <Navbar/>

      <Router>
        <Routes>
            <Route path="/" element= {<HomePage/>} />
            <Route path="/app" element= {<AppBody/>} />
            <Route path="*" element={<Nomatch/>}/>
        </Routes>

      </Router>
      <Footer/>

    </div>
  );
}   

export default App;




// default page when a false url is entred

function Nomatch() {
  return (
    <div>
      <section style={ NomatchStyle }>Ops!!! rong URL :( </section>
    </div>
  )
}

const NomatchStyle = {
  "font-size": "50px",
  "font-weight" : "bold",
  "height": "100vh",
  "display": "flex",
  "justify-content" : "center",
  "align-items" : "center"
}