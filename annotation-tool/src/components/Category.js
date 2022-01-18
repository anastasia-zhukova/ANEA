


import { IoClose } from 'react-icons/io5';

const Category = ({catName, color, catId , datasets, setDatasets, delColor}) => {
    const delCategory = (id) => {
        let keys = Object.keys( datasets[0])
        if(window.confirm(`are you sure you want to delete the category: ${keys[id]}? `)){
            let newData = datasets;
            delete newData[0][keys[id]];
            delColor(catName);
            setDatasets([...newData]);
        }
    }

    return (
        <h5 style={{backgroundColor: color,
            border: `7px solid ${color}`
        }} >{catName} <IoClose className='del-btn' onClick={()=>delCategory(catId)}/> </h5>
    )
}

export default Category
