
import './Menu.css'
import React, { useRef } from 'react'
import { useState } from 'react/cjs/react.development';

const Menu = ({currentCat, datasets, add, change, del,setAdd, setDel, setChange, selectedTxt, setDatasets, currentTerm}) => {

    const [catDisplayed, setCatDisplay] = useState(false);


    let cats = Object.keys(datasets[0]);

    const delTdata = (id, td) => {
        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        let index = newData[0][ keys[id]].indexOf(td);
        newData[0][ keys[id]].splice(index, 1);
        setDel(false)
        setDatasets([...newData]);
        

    }
    const addEntry = (catId, newValue) => {
        console.log(newValue);
        if (newValue.trim().length === 0) return;
        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        newData[0][keys[catId]].push(newValue);
        setAdd(!add);
        setCatDisplay(false)
        setDatasets( [...newData]);


    }
    const changeCat= (oldCat, newCat, value) => {
        delTdata(oldCat, value);
        addEntry(newCat, value)
    }
    // const returnCats = () => {
    //     if (addSelected) {
    //         addSelected = false;
    //         return (cats.map((cat, index) => (<h4 onClick={()=>addEntry(index, selectedTxt) } className='menu-item' key={index}> {cat}</h4>)))
        
    //     }else if (changeSelected) {
    //         changeSelected = false;
    //         return (cats.map((cat, index) => (<h4 onClick={()=>changeCat(currentCat, index, selectedTxt) } className='menu-item' key={index}> {cat}</h4>)))
    //     }
      
    // }
  
    if (catDisplayed) {
        return(
            <div className='menu-cont'>
                <h5>{currentTerm}</h5>
                <div className='items-cont'>
                    <button disabled ={!add} className="menu-item" onClick={()=>(setCatDisplay(!catDisplayed))}>Add to category</button>
                    <button disabled={!change} className="menu-item">Change category</button>
                    <button disabled={!del} className="menu-item" onClick={()=> delTdata(currentCat, currentTerm)}> Delete</button>
                </div>
                <div className="cats-div">
                    {cats.map((cat, index) => (<h4 onClick={()=>addEntry(index, selectedTxt) } className='menu-item' key={index}> {cat}</h4>))}
                </div>
            </div>
        )
    }else 
        return (
            <div className='menu-cont'>
                <h5>{currentTerm}</h5>

                <div className='items-cont'>
                    <button disabled ={!add} className="menu-item" onClick={()=>(setCatDisplay(!catDisplayed))}>Add to category</button>
                    <button disabled={!change} className="menu-item">Change category</button>
                    <button disabled={!del} className="menu-item" onClick={()=> delTdata(currentCat, currentTerm)}> Delete</button>
                </div>
                <div className="cats-div">
                   
                </div>
            </div>
        )
  
}

export default Menu
