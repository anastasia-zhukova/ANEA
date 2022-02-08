
import './Menu.css'

import { useState } from 'react/cjs/react.development';

const Menu = ({currentCat, datasets, add, change, del,setAdd, setDel, setChange, selectedTxt, setDatasets, currentTerm}) => {

    const [catDisplayed, setCatDisplay] = useState(false);
    const [changeSelected, setChngSelect] = useState(false);
   

    let cats = Object.keys(datasets[0]);

    const delTdata = (id, td) => {
        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        let index = newData[0][ keys[id]].indexOf(td);
        newData[0][ keys[id]].splice(index, 1);
        setDel(false)
        setChange(false)

        setDatasets([...newData]);
        

    }
    const addEntry = (catId, newValue) => {
        if (newValue.trim().length === 0) return;
        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        newData[0][keys[catId]].push(newValue);
        setAdd(!add);
        setCatDisplay(false)
        setDatasets( [...newData]);


    }
    const changeCat= (oldCat, newCat, value) => {
        if (value.trim().length === 0) return;
        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        let index = newData[0][ keys[oldCat]].indexOf(value);
        newData[0][ keys[oldCat]].splice(index, 1);


        newData[0][keys[newCat]].push(value);
        setCatDisplay(false)
        setDatasets( [...newData]);
        setChngSelect(false)
        setDel(false)
        setChange(false)
        // delTdata(oldCat, value);
        // addEntry(newCat, value);
        // setCatDisplay(false);
    }

  
    if (catDisplayed) {
        return(
            <div className='menu-cont'>
                <h5>{currentTerm}</h5>
                <div className='items-cont'>
                    <button disabled ={!add} className="menu-item" onClick={()=>(setCatDisplay(!catDisplayed))}>Add to category</button>
                    <button disabled={!change} className="menu-item" onClick={()=>{
                                                            setChngSelect(true);
                                                            setCatDisplay(!catDisplayed)
                                                            }}>Change category</button>
                    <button disabled={!del} className="menu-item" onClick={()=> delTdata(currentCat, currentTerm)}> Delete</button>
                </div>
                <div className="cats-div">
                    {cats.map((cat, index) => (<h4 onClick={()=>{
                        if (changeSelected) {
                            changeCat(currentCat, index, currentTerm);
                            return;
                        }
                        addEntry(index, selectedTxt);
                        
                        }} className='menu-item' key={index}> {cat}</h4>))}
                </div>
            </div>
        )
    }else 
        return (
            <div className='menu-cont'>
                <h5>{currentTerm}</h5>

                <div className='items-cont'>
                    <button disabled ={!add} className="menu-item" onClick={()=>(setCatDisplay(!catDisplayed))}>Add to category</button>
                    <button disabled={!change} className="menu-item" onClick={()=>{
                        setChngSelect(true);
                        setCatDisplay(!catDisplayed);
                    }}>Change category</button>
                    <button disabled={!del} className="menu-item" onClick={()=> delTdata(currentCat, currentTerm)}> Delete</button>
                </div>
                <div className="cats-div">
                   
                </div>
            </div>
        )
  
}

export default Menu
