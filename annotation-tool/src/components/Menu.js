
import './Menu.css'
import React, { useRef } from 'react'

const Menu = () => {
    const listRef = useRef();
    const menuStyle = {

    };
    return (
        <div className='menu'>
            <div className='main-menu'>
                <div className='menu-item'>Add to category</div>   
                <div className='menu-item'>Change category</div>   
                <div className='menu-item'>Delete </div>  
            </div> 
            <div ref={listRef} className="cat-list">
                <h4 className='menu-item'>helloooo</h4>
                <h4 className='menu-item'>helloooo</h4>
                <h4 className='menu-item'>helloooo</h4>
                <h4 className='menu-item'>helloooo</h4>
                <h4 className='menu-item'>helloooo</h4>
                <h4 className='menu-item'>helloooo</h4>
            </div>
        </div>
    )
}

export default Menu
