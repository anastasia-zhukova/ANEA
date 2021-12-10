
import { TiThMenu } from 'react-icons/ti'
import {useState, useEffect} from 'react'
import './Navbar.css'
const Navbar = () => {

    const [toggleMenu, setToggleMenu] = useState(false);
    const [screenWidth, setScreenWidth] = useState(window.innerWidth);

    const toggleNav = () => {
      setToggleMenu(!toggleMenu);
    }
      
    useEffect(() => {
        const changeWidth =() => {
          setScreenWidth(window.innerWidth);
        }
        window.addEventListener('resize', changeWidth);
        return () => {
          window.removeEventListener('resize', changeWidth)
        }
    },[])
    
    return (
        
        <nav className="nav" >
            <div className="brand" >Annotation tool</div>

            
            {(toggleMenu || screenWidth > 800) && ( 
            <ul  className="navItems">
                <li>Home</li>
                <li>About</li>
                <li>Contact Us</li>
                <li onClick={toggleNav}></li>
            </ul>
            )}
            <TiThMenu className="menuBtn" onClick={toggleNav} />
        </nav>
    )
}

export default Navbar


