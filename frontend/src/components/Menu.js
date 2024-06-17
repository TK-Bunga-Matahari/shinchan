import '../index.css';
import { Link } from 'react-router-dom';

const Menu = () => {
    return (
        <div className="bg-gray-100 min-h-screen flex">
            {/* Sidebar */}
            <div className="w-60 bg-white shadow-md px-5 py-7">
                <h1 className="text-xl font-semibold mb-10">Optimization</h1>
                <ul className="space-y-4">
                <li>
                    <a href="#" className="text-gray-700 hover:text-gray-900">Toolbox</a>
                    {/* <link to='/menu' className='text-gray-700 hover:text-gray-900'>Toolbox</link> */}
                </li>
                <li>
                    <a href="#" className="text-gray-700 hover:text-gray-900">Results</a>
                    {/* <link to='/menu/result' className='text-gray-700 hover:text-gray-900'>Results</link> */}
                </li>
                </ul>
            </div>

            {/* Main Content */}
            <div className="flex-grow p-10">
                <div className="max-w-4xl mx-auto">
                <h2 className="text-2xl font-bold mb-6">Optimization Toolbox</h2>
                <div className="grid grid-cols-2 gap-4">
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                    Task Assignment
                    </button>
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                    Portfolio Allocation
                    </button>
                </div>
                </div>
            </div>
        </div>
    );
}

export default Menu;
