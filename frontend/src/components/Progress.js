import '../index.css';
import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Asumsi menggunakan axios untuk fetch data

const Progress = () => {
    const navigate = useNavigate();
    const [inputValue, setInputValue] = useState('');
    const [outputLog, setOutputLog] = useState('');
    

    const handleMenu = () => {
        navigate("/menu");
    }

    const handleResult = () => {
        navigate("/menu/result");
    }

    const handleProgress = () => {
        navigate("/menu/progress");
    }

    const handleInputChange = (e) => {
        setInputValue(e.target.value);
    }

    const handleSubmit = async () => {
        try {
            // Mengirim input ke backend atau memanggil API yang akan mengembalikan isi dari output.log
            const response = await axios.post('../output/output22.log', { input: inputValue });
            setOutputLog(response.data); // Asumsi response data adalah isi dari output.log
        } catch (error) {
            console.error('Error fetching output log:', error);
            setOutputLog('Failed to fetch output log.');
        }
    }

    useEffect(() => {
        handleSubmit(); // Automatically call handleSubmit on component mount to fetch and display the log
    }, []);


    return (
        <div className="bg-gray-100 min-h-screen flex">
            {/* Sidebar */}
            <div className="w-60 bg-white shadow-md px-5 py-7">
                <h1 className="text-xl font-semibold mb-10">Optimization</h1>
                <ul className="space-y-4">
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleMenu}
                        >
                        Toolbox
                        </button>
                    </li>
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleResult}
                        >
                        Results
                        </button>
                    </li>
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleProgress}
                        >
                        Track progress
                        </button>
                    </li>
                </ul>
            </div>

            {/* Main Content */}
            <div className="flex-grow p-10">
            <input
                    type="text"
                    value={inputValue}
                    onChange={handleInputChange}
                    className="border-2 border-gray-300 p-2 rounded-lg"
                    placeholder="Enter your ID here..."
                />
                <button
                    onClick={handleSubmit}
                    className="ml-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                >
                    Submit
                </button>
                {outputLog && (
                    <div className="mt-4 p-4 bg-gray-200 rounded shadow-inner">
                        <pre>{outputLog}</pre>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Progress;
