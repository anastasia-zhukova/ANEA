

import React from 'react'
import ReactTooltip from 'react-tooltip';

const Word = ({setCrntCat, setCrntTerm, datasets, setDatasets, word, color, category, catId, setDel, del, setChange}) => {
    const wordStyle = {
        color : "white",
        backgroundColor: color,
        padding: "2px 2px",
        textAlign: "center",
        whiteSpace: "nowrap",
        borderRadius: "5px",
        margin: "0 7px",
        cursor: "pointer"
    }

    const delTdata = (id, td) => {

        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        let index = newData[0][ keys[id]].indexOf(td);
        newData[0][ keys[id]].splice(index, 1);

        setDatasets([...newData]);
        

    }
    return (
        <>
            <span 
            data-tip={`category: ${category}`} 
            style={wordStyle} 
            onClick={()=>{setDel(true);
                        setChange(true);
                        setCrntTerm(word.trim());
                        setCrntCat(catId);
            }}
            
            >

                {word}
            </span>
            <ReactTooltip />
        </>

    )



    
}




export default Word
