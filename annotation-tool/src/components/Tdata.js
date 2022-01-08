

import  { useRef } from 'react'
import './Tdata.css'
import { AiOutlineEdit } from 'react-icons/ai';
import { IoClose } from 'react-icons/io5';
import {IoMdDoneAll} from 'react-icons/io';
import  { useState} from 'react';


import {MdOutlineAddBox} from 'react-icons/md'






const Tdata = ({data, id, addCell, delTdata, editData, addEntry}) => {

    

    const [valueSelected, setSelection] = useState(false)
    let inputRef = useRef()
    let newEntryInput = useRef();


    const handelEditField = () => {
        if(!valueSelected){
            setSelection(true);
        }else{
            let inputContent = inputRef.current.value;
            //console.log(inputContent);
            if (inputContent !== data) {
                editData(id, data, inputContent);
            }
            setSelection(false);
        }
        //setSelection(!valueSelected)
    }
    const returnField = (selected, text) => {
        if(addCell){
            //console.log(id);
            return <>
                    <div className='valueContainer'>
                            <input type="text" ref={newEntryInput}  />
                            <MdOutlineAddBox className='addIcon' onClick={()=>{addEntry(id, newEntryInput.current.value); newEntryInput.current.value = ""}}/>
                    </div>
                </>
        }
        if (selected) 
            return <>
                    <input ref={inputRef} autoFocus type='text' onBlur={handelEditField} defaultValue={text} className='tdValue'/>
                    <IoMdDoneAll className='editIcon'  onClick={handelEditField}/>
                    <IoClose className='closeIcon' onClick={()=>(delTdata(id, data))}/>

                </>

        else
            return <>
                <h5 className='tdValue'>{text}</h5>
                <AiOutlineEdit className='editIcon'  onClick={handelEditField}/>
                <IoClose className='closeIcon' onClick={()=>(delTdata(id, data))}/>


            </>
    }
    const returnTd = (td)=>{
        if (td) // to check if the cell has any data, otherwise we dont need to dispalay only the buttons/icons
            return(
                <div className='valueContainer'>
    
                    {returnField(valueSelected, td)}
                </div>

            );
        else
                return "";
        
    }

    return (
        <td >
            {returnTd(data)}

        </td>
    )
}
Tdata.defaultProps = {
    addCell: false, 
}
export default Tdata
