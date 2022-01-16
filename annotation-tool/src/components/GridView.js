import './GridView.css';
import React, { useState ,useEffect, useRef} from 'react';
import TRow from './TRow';
import { AiFillDelete } from 'react-icons/ai';
import {MdOutlineAddBox} from 'react-icons/md'
import Tdata from './Tdata';






const GridView = ({datasets, setDatasets}) => {
    const [headers, setHeaders] = useState([]);
    const [tableData, setTableData] = useState([]);
     //TODO u may need to move this back to the App component

    let catInput = useRef();
    let tableRef = useRef();
    var countRows =1;

//////////////////////////////////Tdata functions ////////////////////////////////
    const delTdata = (id, td) => {

        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        let index = newData[0][ keys[id]].indexOf(td);
        newData[0][ keys[id]].splice(index, 1);

        setDatasets([...newData]);
        

    }


    const editData = (id, td, newVal) => {
        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        let index = newData[0][ keys[id]].indexOf(td);
        newData[0][keys[id]][index] = newVal;
        setDatasets([...newData]);
    }
//////////////////////////////////App data Functions ////////////////////////////
    const addEntry = (catId, newValue) => {//////////////TODO empty field after adding

        if (newValue.trim().length === 0) return;
        let keys = Object.keys( datasets[0]);
        let newData = datasets;
        newData[0][keys[catId]].push(newValue);
        setDatasets( [...newData]);

    }
    const getHeader =(datasets) => {
        console.log(datasets);
        let headers = [];
        let id = 0;
        var map = datasets.map((td) => Object.entries(td));
            //console.log(Array.isArray(map));
        //}
        //console.log(datasets.length);
        if(map.length)
            map[0].forEach(row => {
                
                headers = [...headers, {id:id++, hName: row[0]}];
            });
        // for (let i = 0; i < map[0].length; i++) {
        //     headers = [...headers, map[0][i]];
            
            
        // }    
        //console.log(map);
        setHeaders(headers);
    
    }
    const getRows = (datasets) => {
        let data = [];
            var map = datasets.map((td) => Object.entries(td));
            
            let array = map[0];

            var labelValues = array.map(row=> (row[1]));
            var maxArray= labelValues[0].length;
            //console.log(maxArray);
            var k=0
            for (let i = 0; i < labelValues.length; i++) {
                if (labelValues[i].length > maxArray) {
                    maxArray = labelValues[i].length;
                    k=i;
                }
                
            }
            data = labelValues[k].map((_, colIndex) => labelValues.map(row => row[colIndex]));
     
            //console.log(data)
            setTableData(data);
    }




    const delCategory = (id) => {
        let keys = Object.keys( datasets[0])
        if(window.confirm(`are you sure you want to delete the category: ${keys[id]}? `)){
            let newData = datasets;
            delete newData[0][keys[id]];
            setDatasets([...newData]);
        }
    }

    const  addCategory = () => {
        let newData = datasets;
        
        let catName = catInput.current.value.trim();
        if (catName.length === 0) return;

        newData[0][catName] = [];
        setDatasets([...newData]);//TODO che
        //console.log(newData);
        catInput.current.value ="";
        tableRef.current.scrollLeft = tableRef.current.scrollWidth;

        //alert("Category added succesfuly :)")

        
        //console.log(catName);

    }
    useEffect(() => {
        if(datasets.length){
            getHeader(datasets);

            getRows(datasets);
        }

 
  
     },[datasets]);


        return(
            <div className='body-container'>
                {/* <button onClick={() => {getHeader(datasets); getRows(datasets); }} className='temp-btn'>Show dataset</button> */}
                <div className='tableContainer' ref={tableRef}>
                    <table >
                        <thead>
                            <tr>
                                <th>
                                    <div className='valueContainer'>
                                    <input type="text"  ref={catInput} />
                                    <MdOutlineAddBox className='addIcon' onClick={addCategory}/>
                                    </div>
                                </th>

                                {
                                    headers.map((head)=><th key={head.id }>{head.hName}<AiFillDelete id={head.id} className='delIcon' onClick={() => delCategory(head.id)}/> </th>)
                                }
                            </tr>
                        </thead>
                        <tbody>
                            <tr className='addContainer'>
                                <td>--</td>
                                {
                                    headers.map((_, index) => {
                                        return(
                                            
                                               
                                            <Tdata data =" " addEntry={addEntry} id={index} addCell ={true} key={index} />
                                            
                                        )
                                    })
                                }
                            </tr>
                            {
                                
                                tableData.map((row)=>(<TRow editData={editData} delTdata={delTdata} key = {countRows++} count= {countRows} row ={row} />))
                            }
                    
                        </tbody>
                    </table>
                </div>
            </div>
        )
}


export default GridView
