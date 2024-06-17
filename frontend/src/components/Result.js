import React from 'react';
import '../index.css';
import { useState } from 'react';
import image from '../output/similarity_all.png';

const Result = () => {

    // Dummy image URL (replace with actual backend URL)
    const imageUrl = "https://via.placeholder.com/500";

    return (
        <div className="bg-gray-100 min-h-screen flex">
            {/* Sidebar */}
            <div className="w-60 bg-white shadow-md px-5 py-7">
                <h1 className="text-xl font-semibold mb-10">Optimization</h1>
                <ul className="space-y-4">
                <li><a href="#" className="text-gray-700 hover:text-gray-900">Toolbox</a></li>
                <li><a href="#" className="text-gray-700 hover:text-gray-900">Results</a></li>
                </ul>
            </div>

            {/* Main Content */}
            <div className="flex-grow p-10">
                <div className="max-w-4xl mx-auto">
                <h2 className="text-2xl font-bold mb-6">Results</h2>
                <ResultsComponent />
                </div>
            </div>
        </div>
    );
};

function ResultsComponent() {
    const [showTable, setShowTable] = useState(false);
    const [showImage, setShowImage] = useState(false);

    return (
        <div className="space-y-8">
            {/* Card 1: Table */}
            <div className="bg-white p-4 shadow-lg rounded-lg">
            <h3 className="text-lg font-semibold mb-4">Employee Details</h3>
            <div className="overflow-x-auto">
                {/* Placeholder for table */}
                <table className="min-w-full table-auto">
                <thead className="bg-gray-200">
                    <tr>
                    <th className="px-4 py-2">Employee</th>
                    <th className="px-4 py-2">Assigned Task</th>
                    <th className="px-4 py-2">Company</th>
                    <th className="px-4 py-2">Assigned Story Point</th>
                    <th className="px-4 py-2">Wasted Story Point</th>
                    <th className="px-4 py-2">Similarity Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                    <td className="border px-4 py-2">Talent 101</td>
                    <td className="border px-4 py-2">T31, T54, T65</td>
                    <td className="border px-4 py-2">C3</td>
                    <td className="border px-4 py-2">18</td>
                    <td className="border px-4 py-2">2</td>
                    <td className="border px-4 py-2">80%</td>
                    </tr>
                    {/* Additional rows can be added here */}
                </tbody>
                </table>
            </div>
            </div>
    
            {/* Card 2: Image */}
            <div className="bg-white p-4 shadow-lg rounded-lg">
            <h3 className="text-lg font-semibold mb-4">Plot Visualization</h3>
            <div className="flex justify-center items-center h-64">
                {/* Placeholder for image */}
                <img src={ image } alt="Placeholder" className="max-h-full" />
            </div>
            </div>
        </div>
    );
}

export default Result;
