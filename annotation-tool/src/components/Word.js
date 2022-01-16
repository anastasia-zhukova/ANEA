

import React from 'react'
import ReactTooltip from 'react-tooltip';

const Word = ({word, color, category}) => {
    let catName = category.slice(0, 3);
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

    return (
        <>
            <span data-tip={`category: ${category}`} style={wordStyle} >{word}</span>
            <ReactTooltip />
        </>

    )



    
}




export default Word
